using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Padding {

    /// <summary>1次元パディング基底クラス</summary>
    public abstract class Padding1D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>パディング左幅</summary>
        public uint PadLeft { private set; get; }

        /// <summary>パディング右幅</summary>
        public uint PadRight { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(PadLeft)} = {PadLeft} {nameof(PadRight)} = {PadRight}";

        /// <summary>コンストラクタ</summary>
        public Padding1D(uint channels, uint pad_left, uint pad_right) {
            if (channels < 1) {
                throw new ArgumentException(nameof(channels));
            }

            this.Channels = channels;
            this.PadLeft = pad_left;
            this.PadRight = pad_right;
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint inwidth = (args[2] as uint?).Value;
            uint batches = (args[3] as uint?).Value;

            uint outwidth = inwidth + PadLeft + PadRight;
            
            for (uint th = 0; th < batches; th++) {
                Kernel.Execute((Channels, outwidth),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * Channels * inwidth), 
                    outmap.ElementPtr(th * Channels * outwidth),
                    inwidth, outwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inwidth) || inwidth < 1) {
                throw new ArgumentException($"{nameof(args)}[2]");
            }

            if (!(args[3] is uint batches) || batches < 1) {
                throw new ArgumentException($"{nameof(args)}[3]");
            }

            uint outwidth = inwidth + PadLeft + PadRight;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * batches) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }
            
            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * batches) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }
        }
    }
}
