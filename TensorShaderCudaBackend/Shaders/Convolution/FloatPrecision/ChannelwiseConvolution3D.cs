using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution.FloatPrecision {

    /// <summary>チャネルごとの3次元畳み込み</summary>
    public sealed class ChannelwiseConvolution3D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelDepth { private set; get; }

        /// <summary>実行あたりの積数(2^30=1073741824‬)</summary>
        public static ulong MulPerExecute => 0x40000000;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(KernelDepth)} = {KernelDepth}";

        /// <summary>コンストラクタ</summary>
        public ChannelwiseConvolution3D(uint channels, uint kwidth, uint kheight, uint kdepth) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException(null, nameof(channels));
            }
            if (!Limits.CheckKernelSize(kwidth, kheight, kdepth)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}, {nameof(kdepth)}");
            }

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;

            string code = $@"

            {Defines.Float.Fma}

            __global__ void chwise_convolution_3d(const float* __restrict__ inmap, float* __restrict__ outmap, const float* __restrict__ filter,
                                                  unsigned int oy_offset, unsigned int oz,
                                                  unsigned int inwidth, unsigned int outwidth,
                                                  unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                if(ch >= {Channels}){{
                    return;
                }}

                float uv = 0.0;

                unsigned int filter_idx = ch;

                for(unsigned int kz = 0, iz = oz; kz < {KernelDepth}; kz++, iz++){{
                    for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{

                        unsigned int inmap_idx = ch + {Channels} * (ox + inwidth * (iy + inheight * iz));

                        for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{
                            float u = inmap[inmap_idx];
                            float v = filter[filter_idx];

                            float_fma(uv, u, v);

                            filter_idx += {Channels};
                            inmap_idx += {Channels};
                        }}
                    }}
                }}

                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));

                outmap[outmap_idx] = uv;
            }}";

            this.Kernel = new Kernel(code, "chwise_convolution_3d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint inheight = (args[4] as uint?).Value;
            uint indepth = (args[5] as uint?).Value;
            uint batches = (args[6] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            ulong mul_per_line = (ulong)Channels * KernelWidth * KernelHeight * KernelDepth * outwidth;

            uint lines_per_execute = (uint)(MulPerExecute / mul_per_line + 1);

            for (uint th = 0; th < batches; th++) {
                for (uint oz = 0; oz < outdepth; oz++) {
                    for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                        uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                        Kernel.Execute(
                            indexes: (Channels, outwidth, lines),
                            block: (Kernel.DefaultBlockSize(Channels), 1, 1),
                            dynamic_shared_memory_bytes: 0, stream,
                            inmap.ElementPtr(th * Channels * inwidth * inheight * indepth),
                            outmap.ElementPtr(th * Channels * outwidth * outheight * outdepth),
                            filter,
                            oy_offset, oz,
                            inwidth, outwidth, inheight, outheight
                        );
                    }
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 7) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[3] is not uint inwidth || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[4] is not uint inheight || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (args[5] is not uint indepth || !Limits.CheckDepth(indepth, KernelDepth)) {
                throw new ArgumentException(nameof(indepth));
            }

            if (args[6] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < Channels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < Channels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < Channels * KernelWidth * KernelHeight * KernelDepth) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
