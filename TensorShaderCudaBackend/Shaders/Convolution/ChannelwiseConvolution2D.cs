using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>チャネルごとの2次元畳み込み</summary>
    public sealed class ChannelwiseConvolution2D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>実行あたりの積数(2^25=33554432‬)</summary>
        public static uint MulPerExecute => 0x2000000;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight}";

        /// <summary>コンストラクタ</summary>
        public ChannelwiseConvolution2D(uint channels, uint kwidth, uint kheight) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException(nameof(channels));
            }
            if (!Limits.CheckKernelSize(kwidth, kheight)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}");
            }

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;

            string code = $@"

            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo -= (hi - tmp) - val;
            }}

            __global__ void chwise_convolution_2d(float *inmap, float *outmap, float *filter, 
                                                  unsigned int oy_offset,
                                                  unsigned int inwidth, unsigned int outwidth) {{

                unsigned int ch = {Defines.IndexX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                if(ch >= {Channels}){{
                    return;
                }}

                float uv_hi = 0.0, uv_lo = 0.0;

                for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{
                    for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{ 

                        unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * iy);
                        unsigned int filter_idx = ch + {Channels} * (kx + {KernelWidth} * ky);
                                             
                        float u = inmap[inmap_idx];
                        float v = filter[filter_idx];

                        floatfloat_add(uv_hi, uv_lo, u * v);
                    }}
                }}

                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * oy);

                outmap[outmap_idx] = uv_hi + uv_lo;
            }}";

            this.Kernel = new Kernel(code, "chwise_convolution_2d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint inheight = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;

            uint mul_per_line = Channels * KernelWidth * KernelHeight * outwidth;

            uint lines_per_execute = MulPerExecute / mul_per_line + 1;

            for (uint th = 0; th < batches; th++) {
                for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                    uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                    Kernel.Execute(
                        indexes: (Channels, outwidth, lines),
                        block: (Kernel.DefaultBlockSize(Channels), 1, 1),
                        dynamic_shared_memory_bytes: 0, stream,
                        inmap.ElementPtr(th * Channels * inwidth * inheight),
                        outmap.ElementPtr(th * Channels * outwidth * outheight),
                        filter,
                        oy_offset,
                        inwidth, outwidth
                    );
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 6) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint inheight) || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[5] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < Channels * inwidth * inheight * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < Channels * KernelWidth * KernelHeight) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
