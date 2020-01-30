using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Quaternion.Convolution {

    /// <summary>全結合</summary>
    public sealed class Dense : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>Xスレッド数</summary>
        private uint ThreadsX { set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public Dense(uint inchannels, uint outchannels, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 4, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels / 4;
            this.OutChannels = outchannels / 4;
            this.GradMode = gradmode;

            this.ThreadsX = Kernel.DefaultBlockSize(OutChannels);

            string code = $@"

            {Defines.CtorFloat4}
            {Defines.FloatFloatFma}
            {Defines.FloatFloatFms}
            {Defines.FloatFloatHiLoAdd}
            {Defines.Quaternion.Mul}
            {Defines.Quaternion.MulGrad}
            {Defines.StoreFloatSharedMemory(elemsize: 4, InChannels, ThreadsX)}

            __global__ void quaternion_dense(const float4* __restrict__ inmap, float4* __restrict__ outmap, const float4* __restrict__ filter) {{

                unsigned int outch = {Defines.IndexX}, th = {Defines.BlockIndexY};
                unsigned int tid = {Defines.ThreadIdX};

                unsigned int inmap_offset = {InChannels} * th;
                inmap += inmap_offset;

                unsigned int outmap_offset = {OutChannels} * th;
                outmap += outmap_offset;

                __shared__ float4 us[{InChannels}];
                float4 uv_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), uv_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);

                unsigned int filter_idx = outch;

                store_smem(inmap, us, tid);

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                    #pragma unroll 8
                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                        float4 u = us[inch];
                        float4 v = filter[filter_idx];

                        {(GradMode ? "quaternion_mulgrad" : "quaternion_mul")}(uv_hi, uv_lo, u, v);

                        filter_idx += {OutChannels};
                    }}

                    outmap[outch] = ctor_float4(uv_hi.x + uv_lo.x, uv_hi.y + uv_lo.y, uv_hi.z + uv_lo.z, uv_hi.w + uv_lo.w);
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "quaternion_dense");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint batches = (args[3] as uint?).Value;

            CudaArray<float> transpose_filter =
                CudaArrayReserver<float>.Request(stream, filter.DeviceID, index: 0, InChannels * OutChannels * 4);

            TransposeQuaternionKernelChannel(InChannels * 4, OutChannels * 4, 1u, filter, transpose_filter, stream);

            Kernel.Execute(
                indexes: (OutChannels, batches),
                block: (ThreadsX, 1),
                dynamic_shared_memory_bytes: 0, stream,
                inmap,
                outmap,
                transpose_filter
            );
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * batches * 4) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * batches * 4) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
