using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Trimming {

    /// <summary>1次元トリミング</summary>
    public sealed class Trimming1D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>トリミング左幅</summary>
        public uint TrimLeft { private set; get; }

        /// <summary>トリミング右幅</summary>
        public uint TrimRight { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(TrimLeft)} = {TrimLeft} {nameof(TrimRight)} = {TrimRight}";

        /// <summary>コンストラクタ</summary>
        public Trimming1D(uint channels, uint trim_left, uint trim_right) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException(nameof(channels));
            }

            this.Channels = channels;
            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;

            string code = $@"

            __global__ void trimming_1d(float *inmap, float *outmap,
                                       unsigned int outwidth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY};

                if (ch >= {Channels} || ox >= outwidth) {{
                    return;
                }}

                unsigned int ix = ox + {TrimLeft};
                unsigned int inmap_idx = ch + {Channels} * ix;
                unsigned int outmap_idx = ch + {Channels} * ox;

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "trimming_1d");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint outwidth = (args[2] as uint?).Value;
            uint batches = (args[3] as uint?).Value;

            uint inwidth = outwidth + TrimLeft + TrimRight;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, outwidth),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * Channels * inwidth),
                    outmap.ElementPtr(th * Channels * outwidth),
                    outwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint outwidth) || !Limits.CheckWidth(outwidth)) {
                throw new ArgumentException(nameof(outwidth));
            }

            if (!(args[3] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint inwidth = outwidth + TrimLeft + TrimRight;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * batches) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
