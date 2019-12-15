using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Trimming {

    /// <summary>2次元トリミング</summary>
    public sealed class Trimming2D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>トリミング左幅</summary>
        public uint TrimLeft { private set; get; }

        /// <summary>トリミング右幅</summary>
        public uint TrimRight { private set; get; }

        /// <summary>トリミング上幅</summary>
        public uint TrimTop { private set; get; }

        /// <summary>トリミング下幅</summary>
        public uint TrimBottom { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(TrimLeft)} = {TrimLeft} {nameof(TrimRight)} = {TrimRight} " + 
            $"{nameof(TrimTop)} = {TrimTop} {nameof(TrimBottom)} = {TrimBottom}";

        /// <summary>コンストラクタ</summary>
        public Trimming2D(uint channels, uint trim_left, uint trim_right, uint trim_top, uint trim_bottom) {
            if (channels < 1) {
                throw new ArgumentException(nameof(channels));
            }

            this.Channels = channels;
            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;
            this.TrimTop = trim_top;
            this.TrimBottom = trim_bottom;

            string code = $@"

            __global__ void trimming2d(float *inmap, float *outmap, 
                                       unsigned int inwidth, unsigned int outwidth, 
                                       unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                unsigned int ix = ox + {TrimLeft};
                unsigned int iy = oy + {TrimTop};
                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * iy);
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * oy);

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "trimming2d");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint outwidth = (args[2] as uint?).Value;
            uint outheight = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;
            
            uint inwidth = outwidth + TrimLeft + TrimRight;
            uint inheight = outheight + TrimTop + TrimBottom;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute((Channels, outwidth, outheight),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * Channels * inwidth * inheight), 
                    outmap.ElementPtr(th * Channels * outwidth * outheight),
                    inwidth, outwidth, outheight
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint outwidth) || outwidth < 1) {
                throw new ArgumentException($"{nameof(args)}[2]");
            }

            if (!(args[3] is uint outheight) || outheight < 1) {
                throw new ArgumentException($"{nameof(args)}[3]");
            }

            if (!(args[4] is uint batches) || batches < 1) {
                throw new ArgumentException($"{nameof(args)}[4]");
            }

            uint inwidth = outwidth + TrimLeft + TrimRight;
            uint inheight = outheight + TrimTop + TrimBottom;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * inheight * batches) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * batches) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }
        }
    }
}
