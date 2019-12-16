using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Transform {

    /// <summary>チャネル次元を空間方向に展開</summary>
    public sealed class ChannelToSpace2D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>チャネル数</summary>
        public uint Scale { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(OutChannels)} = {OutChannels} {nameof(Scale)} = {Scale}";

        /// <summary>コンストラクタ</summary>
        public ChannelToSpace2D(uint scale, uint outchannels) {
            if (!Limits.CheckChannels(outchannels)) {
                throw new ArgumentException(nameof(outchannels));
            }
            if (scale < 2) {
                throw new ArgumentException(nameof(scale));
            }

            this.InChannels = outchannels * scale * scale;
            this.OutChannels = outchannels;
            this.Scale = scale;

            string code = $@"

            __global__ void channel_to_space_2d(float *inmap, float *outmap, 
                                                unsigned int inwidth, unsigned int inheight) {{

                unsigned int inch = {Defines.IndexX}, ix = {Defines.IndexY}, iy = {Defines.IndexZ};

                if (inch >= {InChannels} || ix >= inwidth || iy >= inheight) {{
                    return;
                }}

                unsigned int outch = inch % {OutChannels};
                unsigned int ox = ix * {Scale} + inch / {OutChannels} % {Scale};
                unsigned int oy = iy * {Scale} + inch / {OutChannels * Scale};
                unsigned int inmap_idx = inch + {InChannels} * (ix + inwidth * iy);
                unsigned int outmap_idx = outch + {OutChannels} * (ox + inwidth * {Scale} * oy);

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "channel_to_space_2d");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint inwidth = (args[2] as uint?).Value;
            uint inheight = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes:(InChannels, inwidth, inheight),
                    dynamic_shared_memory_bytes: 0, 
                    stream,
                    inmap.ElementPtr(th * InChannels * inwidth * inheight), 
                    outmap.ElementPtr(th * InChannels * inwidth * inheight),
                    inwidth, inheight
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inwidth) || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[3] is uint inheight) || !Limits.CheckHeight(inheight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint length = batches * InChannels * inwidth * inheight;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < length) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < length) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
