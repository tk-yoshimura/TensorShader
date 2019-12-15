using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Zoom {

    /// <summary>2次元拡大基底クラス</summary>
    public abstract class Zoom2D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>拡大率</summary>
        public uint Scale => 2;

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} {nameof(Scale)} = {Scale}";

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public Zoom2D(uint channels) {
            if (channels < 1) {
                throw new ArgumentException(nameof(channels));
            }

            this.Channels = channels;
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint inwidth = (args[2] as uint?).Value;
            uint inheight = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;
            
            uint outwidth = inwidth * Scale, outheight = inheight * Scale;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute((Channels, inwidth, inheight),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * Channels * inwidth * inheight), 
                    outmap.ElementPtr(th * Channels * outwidth * outheight),
                    inwidth, outwidth, inheight
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
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

            uint outwidth = inwidth * Scale;
            uint outheight = inheight * Scale;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * inheight * batches) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * batches) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }
        }
    }
}
