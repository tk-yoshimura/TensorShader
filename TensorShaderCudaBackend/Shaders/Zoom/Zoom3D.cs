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
            uint indepth = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;
            
            uint outwidth = inwidth * Scale, outheight = inheight * Scale, outdepth = indepth * Scale;

            for (uint th = 0; th < batches; th++) {
                for(uint iz = 0; iz < indepth; iz++) { 
                    Kernel.Execute((Channels, inwidth, inheight),
                        dynamic_shared_memory_bytes: 0, stream,
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
            if (args == null || args.Length != 6) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inwidth) || inwidth < 1) {
                throw new ArgumentException($"{nameof(args)}[2]");
            }

            if (!(args[3] is uint inheight) || inheight < 1) {
                throw new ArgumentException($"{nameof(args)}[3]");
            }

            if (!(args[4] is uint indepth) || indepth < 1) {
                throw new ArgumentException($"{nameof(args)}[4]");
            }

            if (!(args[5] is uint batches) || batches < 1) {
                throw new ArgumentException($"{nameof(args)}[5]");
            }

            uint outwidth = inwidth * Scale;
            uint outheight = inheight * Scale;
            uint outdepth = indepth * Scale;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }
        }
    }
}
