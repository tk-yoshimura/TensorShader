using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Zoom {

    /// <summary>1次元拡大基底クラス</summary>
    public abstract class Zoom1D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>拡大率</summary>
        public uint Scale => 2;

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} {nameof(Scale)} = {Scale}";

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public Zoom1D(uint channels) {
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
            uint batches = (args[3] as uint?).Value;

            uint outwidth = inwidth * Scale;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, inwidth),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * Channels * inwidth),
                    outmap.ElementPtr(th * Channels * outwidth),
                    inwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inwidth) || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[3] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth * Scale;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * batches) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
