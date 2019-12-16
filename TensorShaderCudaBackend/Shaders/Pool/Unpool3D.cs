using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>3次元逆プール基底クラス</summary>
    public abstract class Unpool3D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>ストライド</summary>
        public uint Stride { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} {nameof(Stride)} = {Stride}";

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public Unpool3D(uint channels, uint stride) {
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
            uint outdepth = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint inwidth = outwidth / Stride, inheight = outheight / Stride, indepth = outdepth / Stride;

            if(outwidth % Stride != 0 || outheight % Stride != 0 || outdepth % Stride != 0) { 
                outmap.ZerosetAsync(stream, Channels * outwidth * outheight * outdepth * batches);
            }

            for (uint th = 0; th < batches; th++) {
                for(uint iz = 0; iz < indepth; iz++) { 
                    Kernel.Execute(
                        indexes:(Channels, inwidth, inheight),
                        dynamic_shared_memory_bytes: 0, 
                        stream,
                        inmap.ElementPtr(th * Channels * inwidth * inheight * indepth), 
                        outmap.ElementPtr(th * Channels * outwidth * outheight * outdepth),
                        iz,
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

            if (!(args[2] is uint outwidth) || !Limits.CheckWidth(outwidth, Stride)) {
                throw new ArgumentException(nameof(outwidth));
            }

            if (!(args[3] is uint outheight) || !Limits.CheckHeight(outheight, Stride)) {
                throw new ArgumentException(nameof(outheight));
            }

            if (!(args[4] is uint outdepth) || !Limits.CheckDepth(outdepth, Stride)) {
                throw new ArgumentException(nameof(outdepth));
            }

            if (!(args[5] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint inwidth = outwidth / Stride;
            uint inheight = outheight / Stride;
            uint indepth = outdepth / Stride;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
