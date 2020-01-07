using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Transform {

    /// <summary>ColumnToImage変換</summary>
    public class ColumnToImage2D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight}";

        /// <summary>コンストラクタ</summary>
        public ColumnToImage2D(uint channels, uint kwidth, uint kheight) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException($"{nameof(channels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}");
            }

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;

            string code = $@"

            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo -= (hi - tmp) - val;
            }}

            __global__ void column_to_image_2d(float *inmap, float *outmap, 
                                               unsigned int inwidth, unsigned int outwidth, 
                                               unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if(ch >= {Channels} || ox >= outwidth || oy >= outheight){{
                    return;
                }}

                float hi = 0.0, lo = 0.0;

                for(unsigned int ky = 0, iy = oy - {KernelHeight - 1}; ky < {KernelHeight}; ky++, iy++){{ 
                    if(iy >= inheight){{
                        continue;
                    }}

                    for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{ 
                        if(ix >= inwidth){{
                            continue;
                        }}

                        unsigned int inmap_idx = ({KernelWidth - 1} - kx) + {KernelWidth} * (({KernelHeight - 1} - ky) + {KernelHeight}
                                               * (ch + {Channels} * (ix + inwidth * iy)));

                        float val = inmap[inmap_idx];

                        floatfloat_add(hi, lo, val);
                    }}
                }}

                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * oy);
                outmap[outmap_idx] = hi + lo;
            }}";

            this.Kernel = new Kernel(code, "column_to_image_2d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;

            uint inwidth = (args[2] as uint?).Value;
            uint inheight = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, outwidth, outheight),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * KernelWidth * KernelHeight * Channels * inwidth * inheight),
                    outmap.ElementPtr(th * Channels * outwidth * outheight),
                    inwidth, outwidth, inheight, outheight
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inwidth) || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[3] is uint inheight) || !Limits.CheckHeight(inheight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;

            if (!(args[0] is CudaArray<float> inmap)
                || inmap.Length < KernelWidth * KernelHeight * Channels * inwidth * inheight * batches) {

                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * batches) {

                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
