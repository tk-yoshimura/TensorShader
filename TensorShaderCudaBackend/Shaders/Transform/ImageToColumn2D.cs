using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Transform {

    /// <summary>ImageToColumn変換</summary>
    public class ImageToColumn2D : Shader {

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
        public ImageToColumn2D(uint channels, uint kwidth, uint kheight) {
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

            __global__ void image_to_column_2d(float *inmap, float *outmap, 
                                               unsigned int inwidth, unsigned int outwidth, 
                                               unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if(ch >= {Channels} || ox >= outwidth || oy >= outheight){{
                    return;
                }}

                unsigned int outmap_idx = {KernelWidth * KernelHeight} * (ch + {Channels} * (ox + outwidth * oy));

                for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{
                    for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{ 

                        unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * iy);

                        outmap[outmap_idx] = inmap[inmap_idx];
                        outmap_idx++;
                    }}
                }}
            }}";

            this.Kernel = new Kernel(code, "image_to_column_2d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;

            uint inwidth = (args[2] as uint?).Value;
            uint inheight = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, outwidth, outheight),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * Channels * inwidth * inheight),
                    outmap.ElementPtr(th * KernelWidth * KernelHeight * Channels * outwidth * outheight),
                    inwidth, outwidth, outheight
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[3] is uint inheight) || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * inheight * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap)
                || outmap.Length < KernelWidth * KernelHeight * Channels * outwidth * outheight * batches) {

                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
