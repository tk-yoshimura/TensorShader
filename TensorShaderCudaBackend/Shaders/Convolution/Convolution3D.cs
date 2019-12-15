using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>3次元畳み込み</summary>
    public sealed class Convolution3D : Shader {
        private readonly Transpose.TransposeKernelChannel transpose;

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelDepth { private set; get; }

        /// <summary>実行あたりの積数(2^25=33554432‬)</summary>
        public static uint MulPerExecute => 0x2000000;
                
        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(KernelDepth)} = {KernelDepth}";
        
        /// <summary>コンストラクタ</summary>
        public Convolution3D(uint inchannels, uint outchannels, uint kwidth, uint kheight, uint kdepth) { 
            if (!Limits.CheckChannels(inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight, kdepth)) { 
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}, {nameof(kdepth)}");
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.transpose = new Transpose.TransposeKernelChannel(inchannels, outchannels);

            string code = $@"

            __global__ void convolution_3d(const float* __restrict__ inmap, float *outmap, float *filter, 
                                           unsigned int oy_offset, unsigned int oz,
                                           unsigned int inwidth, unsigned int outwidth, 
                                           unsigned int inheight, unsigned int outheight) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                __shared__ float us[{InChannels}];
                float uv_hi = 0, uv_lo = 0;

                for(unsigned int kz = 0, iz = oz; kz < {KernelDepth}; kz++, iz++){{
                    for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{
                        for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{ 

                            unsigned int inmap_idx = {InChannels} * (ix + inwidth * (iy + inheight * iz));
                            unsigned int filter_idx = outch + {InChannels * OutChannels} * (kx + {KernelWidth} * (ky + {KernelHeight} * kz));

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
                }}

                if(outch < {OutChannels}){{
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * (oy + outheight * oz));

                    outmap[outmap_idx] = uv_hi + uv_lo;
                }}
            }}";

            this.Kernel = new Kernel(code, "convolution_3d");
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

            uint mul_per_line = InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * outwidth;

            uint lines_per_execute = MulPerExecute / mul_per_line + 1;

            CudaArray<float> transpose_filter = 
                CudaArrayReserver<float>.Request(stream, filter.DeviceID, index:0, InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth);

            transpose.Execute(stream, filter, transpose_filter, KernelWidth * KernelHeight * KernelDepth);

            for (uint th = 0; th < batches; th++) {
                for(uint oz = 0; oz < outdepth; oz++) { 
                    for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                        uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                        Kernel.Execute((OutChannels, outwidth, lines), (Kernel.DefaultBlockSize(OutChannels), 1, 1),
                            dynamic_shared_memory_bytes: 0, stream,
                            inmap.ElementPtr(th * InChannels * inwidth * inheight * indepth), 
                            outmap.ElementPtr(th * OutChannels * outwidth * outheight * outdepth),
                            transpose_filter,
                            oy_offset, oz,
                            inwidth, outwidth, inheight, outheight
                        );
                    }
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 7) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException($"{nameof(inwidth)}");
            }

            if (!(args[4] is uint inheight) || !Limits.CheckWidth(inheight, KernelHeight)) {
                throw new ArgumentException($"{nameof(inheight)}");
            }

            if (!(args[5] is uint indepth) || !Limits.CheckWidth(indepth, KernelDepth)) {
                throw new ArgumentException($"{nameof(indepth)}");
            }

            if (!(args[6] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException($"{nameof(batches)}");
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException($"{nameof(inmap)}");
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException($"{nameof(outmap)}");
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth) {
                throw new ArgumentException($"{nameof(filter)}");
            }
        }
    }
}
