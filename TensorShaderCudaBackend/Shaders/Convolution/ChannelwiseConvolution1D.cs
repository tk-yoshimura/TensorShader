﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>チャネルごとの1次元畳み込み</summary>
    public sealed class ChannelwiseConvolution1D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(KernelWidth)} = {KernelWidth}";

        /// <summary>コンストラクタ</summary>
        public ChannelwiseConvolution1D(uint channels, uint kwidth) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException(nameof(channels));
            }
            if (!Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(nameof(kwidth));
            }

            this.Channels = channels;
            this.KernelWidth = kwidth;

            string code = $@"

            {Defines.FloatFloatFma}

            __global__ void chwise_convolution_1d(float *inmap, float *outmap, float *filter) {{

                unsigned int ch = {Defines.IndexX};
                unsigned int ox = {Defines.BlockIndexY};

                if(ch >= {Channels}){{
                    return;
                }}

                float uv_hi = 0.0, uv_lo = 0.0;

                unsigned int filter_idx = ch;
                unsigned int inmap_idx = ch + {Channels} * ox;

                for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{
                    float u = inmap[inmap_idx];
                    float v = filter[filter_idx];

                    floatfloat_fma(uv_hi, uv_lo, u, v);

                    filter_idx += {Channels};
                    inmap_idx += {Channels};
                }}

                unsigned int outmap_idx = ch + {Channels} * ox;

                outmap[outmap_idx] = uv_hi + uv_lo;
            }}";

            this.Kernel = new Kernel(code, "chwise_convolution_1d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, outwidth),
                    block: (Kernel.DefaultBlockSize(Channels), 1),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * Channels * inwidth),
                    outmap.ElementPtr(th * Channels * outwidth),
                    filter
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < Channels * KernelWidth) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
