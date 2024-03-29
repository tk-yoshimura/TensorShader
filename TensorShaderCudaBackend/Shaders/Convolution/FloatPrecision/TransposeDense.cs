﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution.FloatPrecision {

    /// <summary>転置全結合</summary>
    public sealed class TransposeDense : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>Xスレッド数</summary>
        private uint ThreadsX { set; get; }

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

            this.ThreadsX = Kernel.DefaultBlockSize(OutChannels);

            string code = $@"

            {Defines.Float.Fma}
            {Defines.StoreFloatSharedMemory(elemsize: 1, InChannels, ThreadsX)}

            __global__ void transpose_dense(const float* __restrict__ inmap, float* __restrict__ outmap, const float* __restrict__ filter) {{

                unsigned int outch = {Defines.IndexX}, th = {Defines.BlockIndexY};
                unsigned int tid = {Defines.ThreadIdX};

                unsigned int inmap_offset = {InChannels} * th;
                inmap += inmap_offset;

                unsigned int outmap_offset = {OutChannels} * th;
                outmap += outmap_offset;

                __shared__ float us[{InChannels}];
                float uv = 0.0;

                unsigned int filter_idx = outch;

                store_smem(inmap, us, tid);

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                    #pragma unroll 8
                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                        float u = us[inch];
                        float v = filter[filter_idx];

                        float_fma(uv, u, v);

                        filter_idx += {OutChannels};
                    }}

                    outmap[outch] = uv;
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "transpose_dense");
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
                block: (ThreadsX, 1),
                dynamic_shared_memory_bytes: 0, stream,
                inmap,
                outmap,
                filter
            );
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 4) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[3] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            if (args[0] is not CudaArray<float> inmap || inmap.Length < InChannels * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < OutChannels * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < InChannels * OutChannels) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
