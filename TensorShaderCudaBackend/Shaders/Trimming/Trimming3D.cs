using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Trimming {

    /// <summary>3次元トリミング</summary>
    public sealed class Trimming3D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>トリミング左幅</summary>
        public uint TrimLeft { private set; get; }

        /// <summary>トリミング右幅</summary>
        public uint TrimRight { private set; get; }

        /// <summary>トリミング上幅</summary>
        public uint TrimTop { private set; get; }

        /// <summary>トリミング下幅</summary>
        public uint TrimBottom { private set; get; }

        /// <summary>トリミング前幅</summary>
        public uint TrimFront { private set; get; }

        /// <summary>トリミング後幅</summary>
        public uint TrimRear { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(TrimLeft)} = {TrimLeft} {nameof(TrimRight)} = {TrimRight} " +
            $"{nameof(TrimTop)} = {TrimTop} {nameof(TrimBottom)} = {TrimBottom} " +
            $"{nameof(TrimFront)} = {TrimFront} {nameof(TrimRear)} = {TrimRear}";


        /// <summary>コンストラクタ</summary>
        public Trimming3D(uint channels, uint trim_left, uint trim_right, uint trim_top, uint trim_bottom, uint trim_front, uint trim_rear) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException(null, nameof(channels));
            }

            this.Channels = channels;
            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;
            this.TrimTop = trim_top;
            this.TrimBottom = trim_bottom;
            this.TrimFront = trim_front;
            this.TrimRear = trim_rear;

            string code = $@"

            __global__ void trimming_3d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                       unsigned int oz,
                                       unsigned int inwidth, unsigned int outwidth,
                                       unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                unsigned int ix = ox + {TrimLeft};
                unsigned int iy = oy + {TrimTop};
                unsigned int iz = oz + {TrimFront};
                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + inheight * iz));
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "trimming_3d");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint outwidth = (args[2] as uint?).Value;
            uint outheight = (args[3] as uint?).Value;
            uint outdepth = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint inwidth = outwidth + TrimLeft + TrimRight;
            uint inheight = outheight + TrimTop + TrimBottom;
            uint indepth = outdepth + TrimFront + TrimRear;

            for (uint th = 0; th < batches; th++) {
                for (uint oz = 0; oz < outdepth; oz++) {
                    Kernel.Execute(
                        indexes: (Channels, outwidth, outheight),
                        dynamic_shared_memory_bytes: 0,
                        stream,
                        inmap.ElementPtr(th * Channels * inwidth * inheight * indepth),
                        outmap.ElementPtr(th * Channels * outwidth * outheight * outdepth),
                        oz,
                        inwidth, outwidth, inheight, outheight
                    );
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 6) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[2] is not uint outwidth || !Limits.CheckWidth(outwidth)) {
                throw new ArgumentException(nameof(outwidth));
            }

            if (args[3] is not uint outheight || !Limits.CheckHeight(outheight)) {
                throw new ArgumentException(nameof(outheight));
            }

            if (args[4] is not uint outdepth || !Limits.CheckDepth(outdepth)) {
                throw new ArgumentException(nameof(outdepth));
            }

            if (args[5] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint inwidth = outwidth + TrimLeft + TrimRight;
            uint inheight = outheight + TrimTop + TrimBottom;
            uint indepth = outdepth + TrimFront + TrimRear;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < Channels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < Channels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
