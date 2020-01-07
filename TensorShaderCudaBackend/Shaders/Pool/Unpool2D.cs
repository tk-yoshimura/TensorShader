using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>2次元逆プール基底クラス</summary>
    public abstract class Unpool2D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>ストライド</summary>
        public uint Stride { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} {nameof(Stride)} = {Stride}";

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public Unpool2D(uint channels, uint stride) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException(nameof(channels));
            }
            if (stride < 2) {
                throw new ArgumentException(nameof(stride));
            }

            this.Channels = channels;
            this.Stride = stride;
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint outwidth = (args[2] as uint?).Value;
            uint outheight = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            uint inwidth = outwidth / Stride, inheight = outheight / Stride;

            if (outwidth % Stride != 0 || outheight % Stride != 0) {
                outmap.ZerosetAsync(stream, Channels * outwidth * outheight * batches);
            }

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, inwidth, inheight),
                    dynamic_shared_memory_bytes: 0,
                    stream,
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

            if (!(args[2] is uint outwidth) || !Limits.CheckWidth(outwidth, Stride)) {
                throw new ArgumentException(nameof(outwidth));
            }

            if (!(args[3] is uint outheight) || !Limits.CheckHeight(outheight, Stride)) {
                throw new ArgumentException(nameof(outheight));
            }

            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint inwidth = outwidth / Stride;
            uint inheight = outheight / Stride;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * inheight * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * batches) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
