using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Transform {

    /// <summary>ColumnToImage変換</summary>
    public class ColumnToImage3D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelDepth { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(KernelDepth)} = {KernelDepth}";

        /// <summary>コンストラクタ</summary>
        public ColumnToImage3D(uint channels, uint kwidth, uint kheight, uint kdepth) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException(null, $"{nameof(channels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight, kdepth)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}, {nameof(kdepth)}");
            }

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;

            string code = $@"

            {Defines.FloatFloat.Add}

            __global__ void column_to_image_3d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                               unsigned int oz,
                                               unsigned int inwidth, unsigned int outwidth,
                                               unsigned int inheight, unsigned int outheight,
                                               unsigned int indepth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if(ch >= {Channels} || ox >= outwidth || oy >= outheight){{
                    return;
                }}

                float hi = 0.0, lo = 0.0;

                for(unsigned int kz = 0, iz = oz - {KernelDepth - 1}; kz < {KernelDepth}; kz++, iz++){{
                    if(iz >= indepth){{
                        continue;
                    }}

                    for(unsigned int ky = 0, iy = oy - {KernelHeight - 1}; ky < {KernelHeight}; ky++, iy++){{
                        if(iy >= inheight){{
                            continue;
                        }}

                        for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{
                            if(ix >= inwidth){{
                                continue;
                            }}

                            unsigned int inmap_idx = ({KernelWidth - 1} - kx) + {KernelWidth} * (({KernelHeight - 1} - ky) + {KernelHeight} * (({KernelDepth - 1} - kz) + {KernelDepth}
                                                   * (ch + {Channels} * (ix + inwidth * (iy + inheight * iz)))));

                            float val = inmap[inmap_idx];

                            floatfloat_add(hi, lo, val);
                        }}
                    }}
                }}

                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));
                outmap[outmap_idx] = hi + lo;
            }}";

            this.Kernel = new Kernel(code, "column_to_image_3d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;

            uint inwidth = (args[2] as uint?).Value;
            uint inheight = (args[3] as uint?).Value;
            uint indepth = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;
            uint outdepth = indepth + KernelDepth - 1;

            for (uint th = 0; th < batches; th++) {
                for (uint oz = 0; oz < outdepth; oz++) {
                    Kernel.Execute(
                        indexes: (Channels, outwidth, outheight),
                        dynamic_shared_memory_bytes: 0, stream,
                        inmap.ElementPtr(th * KernelWidth * KernelHeight * KernelDepth * Channels * inwidth * inheight * indepth),
                        outmap.ElementPtr(th * Channels * outwidth * outheight * outdepth),
                        oz,
                        inwidth, outwidth, inheight, outheight, indepth
                    );
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 6) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[2] is not uint inwidth || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[3] is not uint inheight || !Limits.CheckHeight(inheight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (args[4] is not uint indepth || !Limits.CheckDepth(indepth)) {
                throw new ArgumentException(nameof(indepth));
            }

            if (args[5] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;
            uint outdepth = indepth + KernelDepth - 1;

            if (args[0] is not CudaArray<float> inmap
                || inmap.Length < KernelWidth * KernelHeight * KernelDepth * Channels * inwidth * inheight * indepth * batches) {

                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < Channels * outwidth * outheight * outdepth * batches) {

                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
