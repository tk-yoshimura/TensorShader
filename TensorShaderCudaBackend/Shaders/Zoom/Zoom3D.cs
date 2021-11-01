using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Zoom {

    /// <summary>3次元拡大基底クラス</summary>
    public abstract class Zoom3D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>拡大率</summary>
        public uint Scale => 2;

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} {nameof(Scale)} = {Scale}";

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public Zoom3D(uint channels) {
            if (channels < 1) {
                throw new ArgumentException(null, nameof(channels));
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
            uint indepth = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint outwidth = inwidth * Scale, outheight = inheight * Scale, outdepth = indepth * Scale;

            for (uint th = 0; th < batches; th++) {
                for (uint iz = 0; iz < indepth; iz++) {
                    Kernel.Execute(
                        indexes: (Channels, inwidth, inheight),
                        dynamic_shared_memory_bytes: 0,
                        stream,
                        inmap.ElementPtr(th * Channels * inwidth * inheight * indepth),
                        outmap.ElementPtr(th * Channels * outwidth * outheight * outdepth),
                        iz,
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

            uint outwidth = inwidth * Scale;
            uint outheight = inheight * Scale;
            uint outdepth = indepth * Scale;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < Channels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < Channels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
