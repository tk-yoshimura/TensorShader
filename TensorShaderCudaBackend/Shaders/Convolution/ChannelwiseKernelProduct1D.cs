using System;
using System.Linq;

using static TensorShaderCudaBackend.ArrayManipulation;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>カーネル積</summary>
    public sealed class ChannelwiseKernelProduct1D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>1スレッドで処理する対象ピクセル数</summary>
        private static uint BatchPixels => 16;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(KernelWidth)} = {KernelWidth}";

        /// <summary>コンストラクタ</summary>
        public ChannelwiseKernelProduct1D(uint channels, uint kwidth) {
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
            {Defines.AtomicAdd}

            __global__ void chwise_kernelproduct_1d(const float* __restrict__ inmap, const float* __restrict__ outmap, float* __restrict__ filter,
                                                    unsigned int outwidth) {{

                unsigned int ch = {Defines.IndexX};
                unsigned int ox_offset = {Defines.BlockIndexY} * {BatchPixels};

                if(ch >= {Channels}){{
                    return;
                }}

                for(unsigned int kx = 0; kx < {KernelWidth}; kx++){{
                    unsigned int filter_index = (ch + {Channels} * kx) * 2;

                    float uv_hi = 0.0, uv_lo = 0.0;

                    for(unsigned int ox = ox_offset, ix = ox + kx; ox < ox_offset + {BatchPixels} && ox < outwidth; ox++, ix++){{

                        unsigned int inmap_idx = ch + {Channels} * ix;
                        unsigned int outmap_idx = ch + {Channels} * ox;

                        float u = inmap[inmap_idx];
                        float v = outmap[outmap_idx];

                        floatfloat_fma(uv_hi, uv_lo, u, v);
                    }}

                    floatfloat_atomicadd(filter + filter_index, uv_hi, uv_lo);
                }}
            }}";

            this.Kernel = new Kernel(code, "chwise_kernelproduct_1d");
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

            CudaArray<float> dfloat_filter =
                WorkspaceReserver<float>.Request(stream, inmap.DeviceID, index: 0, Channels * KernelWidth * 2);
            dfloat_filter.ZerosetAsync(stream, Channels * KernelWidth * 2);

            uint xsets = (outwidth + BatchPixels - 1) / BatchPixels;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, xsets),
                    block: (Kernel.DefaultBlockSize(Channels), 1),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * Channels * inwidth),
                    outmap.ElementPtr(th * Channels * outwidth),
                    dfloat_filter,
                    outwidth
                );
            }

            HorizontalAdd(Channels * KernelWidth, dfloat_filter, filter, stream);
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
