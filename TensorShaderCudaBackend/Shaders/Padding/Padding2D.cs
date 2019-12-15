using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Padding {

    /// <summary>2次元パディング基底クラス</summary>
    public abstract class Padding2D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>パディング左幅</summary>
        public uint PadLeft { private set; get; }

        /// <summary>パディング右幅</summary>
        public uint PadRight { private set; get; }

        /// <summary>パディング上幅</summary>
        public uint PadTop { private set; get; }

        /// <summary>パディング下幅</summary>
        public uint PadBottom { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(PadLeft)} = {PadLeft} {nameof(PadRight)} = {PadRight} " + 
            $"{nameof(PadTop)} = {PadTop} {nameof(PadBottom)} = {PadBottom}";

        /// <summary>コンストラクタ</summary>
        public Padding2D(uint channels, uint pad_left, uint pad_right, uint pad_top, uint pad_bottom) {
            if (channels < 1) {
                throw new ArgumentException(nameof(channels));
            }

            this.Channels = channels;
            this.PadLeft = pad_left;
            this.PadRight = pad_right;
            this.PadTop = pad_top;
            this.PadBottom = pad_bottom;
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint inwidth = (args[2] as uint?).Value;
            uint inheight = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            uint outwidth = inwidth + PadLeft + PadRight;
            uint outheight = inheight + PadTop + PadBottom;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute((Channels, outwidth, outheight),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * Channels * inwidth * inheight), 
                    outmap.ElementPtr(th * Channels * outwidth * outheight),
                    inwidth, outwidth, inheight, outheight
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inwidth) || inwidth < 1) {
                throw new ArgumentException($"{nameof(args)}[2]");
            }

            if (!(args[3] is uint inheight) || inheight < 1) {
                throw new ArgumentException($"{nameof(args)}[3]");
            }

            if (!(args[4] is uint batches) || batches < 1) {
                throw new ArgumentException($"{nameof(args)}[4]");
            }

            uint outwidth = inwidth + PadLeft + PadRight;
            uint outheight = inheight + PadTop + PadBottom;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * inheight * batches) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * batches) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }
        }
    }
}
