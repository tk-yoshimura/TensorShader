using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Transform {

    /// <summary>ColumnToImage変換</summary>
    public class ColumnToImage1D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(KernelWidth)} = {KernelWidth}";

        /// <summary>コンストラクタ</summary>
        public ColumnToImage1D(uint channels, uint kwidth) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException($"{nameof(channels)}");
            }
            if (!Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(nameof(kwidth));
            }

            this.Channels = channels;
            this.KernelWidth = kwidth;

            string code = $@"

            {Defines.FloatFloat.Add}

            __global__ void column_to_image_1d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                               unsigned int inwidth, unsigned int outwidth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY};

                if(ch >= {Channels} || ox >= outwidth){{
                    return;
                }}

                float hi = 0.0, lo = 0.0;

                for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{
                    if(ix >= inwidth){{
                        continue;
                    }}

                    unsigned int inmap_idx = ({KernelWidth - 1} - kx) + {KernelWidth} * (ch + {Channels} * ix);

                    float val = inmap[inmap_idx];

                    floatfloat_add(hi, lo, val);
                }}

                unsigned int outmap_idx = ch + {Channels} * ox;
                outmap[outmap_idx] = hi + lo;
            }}";

            this.Kernel = new Kernel(code, "column_to_image_1d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;

            uint inwidth = (args[2] as uint?).Value;
            uint batches = (args[3] as uint?).Value;

            uint outwidth = inwidth + KernelWidth - 1;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, outwidth),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * KernelWidth * Channels * inwidth),
                    outmap.ElementPtr(th * Channels * outwidth),
                    inwidth, outwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inwidth) || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[3] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + KernelWidth - 1;

            if (!(args[0] is CudaArray<float> inmap)
                || inmap.Length < KernelWidth * Channels * inwidth * batches) {

                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * batches) {

                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
