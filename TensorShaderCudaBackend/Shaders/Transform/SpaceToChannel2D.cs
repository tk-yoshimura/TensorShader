using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Transform {

    /// <summary>空間方向をチャネル次元に展開</summary>
    public sealed class SpaceToChannel2D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>チャネル数</summary>
        public uint Scale { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(Scale)} = {Scale}";

        /// <summary>コンストラクタ</summary>
        public SpaceToChannel2D(uint scale, uint inchannels) {
            if (inchannels < 1) {
                throw new ArgumentException(nameof(inchannels));
            }

            if (scale < 2) {
                throw new ArgumentException(nameof(scale));
            }

            this.InChannels = inchannels;
            this.OutChannels = inchannels * scale * scale;
            this.Scale = scale;

            string code = $@"

            __global__ void space_to_channel_2d(float *inmap, float *outmap, 
                                                unsigned int outwidth, unsigned int outheight) {{

                unsigned int outch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (outch >= {OutChannels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                unsigned int inch = outch % {InChannels};
                unsigned int ix = ox * {Scale} + outch / {InChannels} % {Scale};
                unsigned int iy = oy * {Scale} + outch / {InChannels * Scale};
                unsigned int inmap_idx = inch + {InChannels} * (ix + outwidth * {Scale} * iy);
                unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * oy);

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "space_to_channel_2d");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint outwidth = (args[2] as uint?).Value;
            uint outheight = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute((OutChannels, outwidth, outheight),
                    dynamic_shared_memory_bytes: 0, stream,
                    inmap.ElementPtr(th * OutChannels * outwidth * outheight), 
                    outmap.ElementPtr(th * OutChannels * outwidth * outheight),
                    outwidth, outheight
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

            uint length = batches * OutChannels * outwidth * outheight;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < length) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < length) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }
        }
    }
}
