using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Complex.Convolution {

    /// <summary>全結合</summary>
    public sealed class Dense : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public Dense(uint inchannels, uint outchannels, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 2, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels / 2;
            this.OutChannels = outchannels / 2;
            this.GradMode = gradmode;

            string code = $@"

            {Defines.CtorFloat2}
            {Defines.FloatFloatAdd}
            {Defines.FloatFloatSub}
            {Defines.Complex.Mul}
            {Defines.Complex.MulGrad}
            {Defines.StoreSharedMemory("float2", InChannels)}

            __global__ void complex_dense(float2 *inmap, float2 *outmap, float2 *filter) {{

                unsigned int outch = {Defines.IndexX}, th = {Defines.BlockIndexY};
                unsigned int tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};

                unsigned int inmap_offset = {InChannels} * th;
                inmap += inmap_offset;

                unsigned int outmap_offset = {OutChannels} * th;
                outmap += outmap_offset;

                __shared__ float2 us[{InChannels}];
                float2 vu_hi = ctor_float2(0.0, 0.0), vu_lo = ctor_float2(0.0, 0.0);

                unsigned int filter_idx = outch;

                store_smem(inmap, us, tid, threads);

                if(outch < {OutChannels}){{
                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                        float2 u = us[inch];
                        float2 v = filter[filter_idx];

                        {(GradMode ? "complex_mulgrad" : "complex_mul")}(vu_hi, vu_lo, v, u);

                        filter_idx += {OutChannels};
                    }}

                    outmap[outch] = ctor_float2(vu_hi.x + vu_lo.x, vu_hi.y + vu_lo.y);
                }}
            }}";

            this.Kernel = new Kernel(code, "complex_dense");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint batches = (args[3] as uint?).Value;

            CudaArray<float> transpose_filter =
                CudaArrayReserver<float>.Request(stream, filter.DeviceID, index: 0, InChannels * OutChannels * 2);

            TransposeComplexKernelChannel(InChannels * 2, OutChannels * 2, 1u, filter, transpose_filter, stream);

            Kernel.Execute(
                indexes: (OutChannels, batches),
                block: (Kernel.DefaultBlockSize(OutChannels), 1),
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

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * batches * 2) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * batches * 2) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * 2) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
