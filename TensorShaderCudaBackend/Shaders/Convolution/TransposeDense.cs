using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>転置全結合</summary>
    public sealed class TransposeDense : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels}";

        /// <summary>コンストラクタ</summary>
        public TransposeDense(uint inchannels, uint outchannels) {
            if (!Limits.CheckChannels(inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;

            string code = $@"

            {Defines.FloatFloatAdd}
            {Defines.StoreSharedMemory("float", InChannels)}

            __global__ void transpose_dense(float *inmap, float *outmap, float *filter) {{

                unsigned int outch = {Defines.IndexX}, th = {Defines.BlockIndexY};
                unsigned int tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};

                unsigned int inmap_offset = {InChannels} * th;
                inmap += inmap_offset;

                unsigned int outmap_offset = {OutChannels} * th;
                outmap += outmap_offset;

                __shared__ float us[{InChannels}];
                float uv_hi = 0.0, uv_lo = 0.0;

                unsigned int filter_idx = outch;

                store_smem(inmap, us, tid, threads);

                if(outch < {OutChannels}){{

                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                        float u = us[inch];
                        float v = filter[filter_idx];

                        floatfloat_add(uv_hi, uv_lo, u * v);

                        filter_idx += {OutChannels};
                    }}

                    outmap[outch] = uv_hi + uv_lo;
                }}
            }}";

            this.Kernel = new Kernel(code, "transpose_dense");
            this.Kernel.SetCacheAllocationFromUsageSharedMemory(InChannels * 4);
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint batches = (args[3] as uint?).Value;

            Kernel.Execute(
                indexes: (OutChannels, batches),
                block: (Kernel.DefaultBlockSize(OutChannels), 1),
                dynamic_shared_memory_bytes: 0, stream,
                inmap,
                outmap,
                filter
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

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
