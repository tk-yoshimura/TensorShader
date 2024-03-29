﻿using System;
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
            if (!Limits.CheckChannels(inchannels)) {
                throw new ArgumentException(null, nameof(inchannels));
            }
            if (scale < 2) {
                throw new ArgumentException(null, nameof(scale));
            }

            this.InChannels = inchannels;
            this.OutChannels = inchannels * scale * scale;
            this.Scale = scale;

            string code = $@"

            __global__ void space_to_channel_2d(const float* __restrict__ inmap, float* __restrict__ outmap,
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
                Kernel.Execute(
                    indexes: (OutChannels, outwidth, outheight),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * OutChannels * outwidth * outheight),
                    outmap.ElementPtr(th * OutChannels * outwidth * outheight),
                    outwidth, outheight
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 5) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[2] is not uint outwidth || !Limits.CheckWidth(outwidth)) {
                throw new ArgumentException(nameof(outwidth));
            }

            if (args[3] is not uint outheight || !Limits.CheckHeight(outheight)) {
                throw new ArgumentException(nameof(outheight));
            }

            if (args[4] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint length = batches * OutChannels * outwidth * outheight;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < length) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < length) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
