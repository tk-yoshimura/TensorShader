using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>2次元逆畳み込み</summary>
    public sealed class Deconvolution2D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>実行あたりの積数(2^25=33554432‬)</summary>
        public static uint MulPerExecute => 0x2000000;
                
        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight}";
        
        /// <summary>コンストラクタ</summary>
        public Deconvolution2D(uint inchannels, uint outchannels, uint kwidth, uint kheight) { 
            if (inchannels < 1 || inchannels > Limits.Channels) {
                throw new ArgumentException(nameof(inchannels));
            }
            if (outchannels < 1 || outchannels > Limits.Channels) {
                throw new ArgumentException(nameof(outchannels));
            }
            if ((kwidth & 1) != 1) { 
                throw new ArgumentException(nameof(kwidth));
            }
            if ((kheight & 1) != 1) { 
                throw new ArgumentException(nameof(kheight));
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;

            string code = $@"

            __global__ void deconvolution_2d(const float* __restrict__ inmap, float *outmap, float *filter,
                                             unsigned int oy_offset,
                                             unsigned int inwidth, unsigned int outwidth, 
                                             unsigned int inheight) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                __shared__ float us[{InChannels}];
                float uv_hi = 0, uv_lo = 0;

                for(unsigned int ky = 0, iy = oy - {KernelHeight - 1}; ky < {KernelHeight}; ky++, iy++){{ 
                    if(iy >= inheight){{
                        continue;
                    }}

                    for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{ 
                        if(ix >= inwidth){{
                            continue;
                        }}

                        unsigned int inmap_idx = {InChannels} * (ix + inwidth * iy);
                        unsigned int filter_idx = outch + {InChannels * OutChannels} * 
                                                  (({KernelWidth - 1} - kx) + {KernelWidth} * ({KernelHeight - 1} - ky));

                        for(unsigned int inch = tid; inch < {InChannels}; inch += threads){{
                            us[inch] = inmap[inch + inmap_idx];
                        }}
                        __syncthreads();

                        if(outch < {OutChannels}){{                        
                            for(unsigned int inch = 0; inch < {InChannels}; inch++){{                            
                                float u = us[inch];
                                float v = filter[filter_idx];

                                float uv = u * v;
                                float tmp = uv_hi;
                                uv_hi += uv;
                                uv_lo -= (uv_hi - tmp) - uv;

                                filter_idx += {OutChannels};
                            }}

                        }}
                        __syncthreads();
                    }}
                }}

                if(outch < {OutChannels}){{
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * oy);

                    outmap[outmap_idx] = uv_hi + uv_lo;
                }}
            }}";

            this.Kernel = new Kernel(code, "deconvolution_2d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;
           
            uint outwidth = (args[3] as uint?).Value;
            uint outheight = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint inwidth = outwidth + 1 - KernelWidth;
            uint inheight = outheight + 1 - KernelHeight;

            uint mul_per_line = InChannels * OutChannels * KernelWidth * KernelHeight * outwidth;

            uint lines_per_execute = MulPerExecute / mul_per_line + 1;

            for (uint th = 0; th < batches; th++) {
                for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                    uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                    Kernel.Execute((OutChannels, outwidth, lines), (Kernel.DefaultBlockSize(OutChannels), 1, 1),
                        dynamic_shared_memory_bytes: 0, stream,
                        inmap.ElementPtr(th * InChannels * inwidth * inheight), 
                        outmap.ElementPtr(th * OutChannels * outwidth * outheight),
                        filter,
                        oy_offset,
                        inwidth, outwidth, inheight
                    );
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 6) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint outwidth) || outwidth < KernelWidth) {
                throw new ArgumentException($"{nameof(args)}[3]");
            }

            if (!(args[4] is uint outheight) || outheight < KernelHeight) {
                throw new ArgumentException($"{nameof(args)}[4]");
            }

            if (!(args[5] is uint batches) || batches < 1) {
                throw new ArgumentException($"{nameof(args)}[5]");
            }

            uint inwidth = outwidth + 1 - KernelWidth;
            uint inheight = outheight + 1 - KernelHeight;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * inheight * batches) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * outheight * batches) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight) {
                throw new ArgumentException($"{nameof(args)}[2]");
            }
        }
    }
}
