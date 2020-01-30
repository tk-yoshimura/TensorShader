using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Transform {

    /// <summary>ImageToColumn変換</summary>
    public class ImageToColumn3D : Shader {

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
        public ImageToColumn3D(uint channels, uint kwidth, uint kheight, uint kdepth) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException($"{nameof(channels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight, kdepth)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}, {nameof(kdepth)}");
            }

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;

            string code = $@"

            __global__ void image_to_column_3d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                               unsigned int oz,
                                               unsigned int inwidth, unsigned int outwidth,
                                               unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if(ch >= {Channels} || ox >= outwidth || oy >= outheight){{
                    return;
                }}

                unsigned int outmap_idx = {KernelWidth * KernelHeight * KernelDepth} * (ch + {Channels} * (ox + outwidth * (oy + outheight * oz)));

                for(unsigned int kz = 0, iz = oz; kz < {KernelDepth}; kz++, iz++){{
                    for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{
                        for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{

                            unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + inheight * iz));

                            outmap[outmap_idx] = inmap[inmap_idx];
                            outmap_idx++;
                        }}
                    }}
                }}
            }}";

            this.Kernel = new Kernel(code, "image_to_column_3d");
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

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            for (uint th = 0; th < batches; th++) {
                for (uint oz = 0; oz < outdepth; oz++) {
                    Kernel.Execute(
                        indexes: (Channels, outwidth, outheight),
                        dynamic_shared_memory_bytes: 0, stream,
                        inmap.ElementPtr(th * Channels * inwidth * inheight * indepth),
                        outmap.ElementPtr(th * KernelWidth * KernelHeight * KernelDepth * Channels * outwidth * outheight * outdepth),
                        oz,
                        inwidth, outwidth, inheight, outheight
                    );
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 6) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[3] is uint inheight) || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[4] is uint indepth) || !Limits.CheckDepth(indepth, KernelDepth)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[5] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap)
                || outmap.Length < KernelWidth * KernelHeight * Channels * outwidth * outheight * outdepth * batches) {

                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
