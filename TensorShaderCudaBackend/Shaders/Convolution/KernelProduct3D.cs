using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>カーネル積</summary>
    public sealed class KernelProduct3D : Shader {
        private static ArrayManipulation.HorizontalAdd hadd;

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

        /// <summary>実行あたりの積数(2^24=16777216‬)</summary>
        public static uint MulPerExecute => 0x1000000;

        /// <summary>実行あたりのピクセル数(2^14=16384‬)</summary>
        public static uint PixelsPerExecute => 0x4000;

        /// <summary>1スレッドで処理する対象ピクセル数</summary>
        private static uint BatchPixels => 16;

        /// <summary>ブロックサイズ</summary>
        public (uint x, uint y) BlockSize { private set; get; }
                
        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " + 
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(KernelDepth)} = {KernelDepth}";
        
        /// <summary>コンストラクタ</summary>
        public KernelProduct3D(uint inchannels, uint outchannels, uint kwidth, uint kheight, uint kdepth) { 
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

            this.BlockSize = Kernel.MinimizeGridsBlockSize((InChannels, OutChannels));

            string code = $@"

            __global__ void kernelproduct_3d(float *inmap, float *outmap, float *filter, 
                                             unsigned int oy_offset,
                                             unsigned int oz,
                                             unsigned int xsets,
                                             unsigned int inwidth, unsigned int outwidth, 
                                             unsigned int inheight, unsigned int outheight) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY};
                unsigned int ox_offset = ({Defines.BlockIndexZ} % xsets) * {BatchPixels};
                unsigned int oy = oy_offset + {Defines.BlockIndexZ} / xsets;
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                __shared__ float us[{BlockSize.x}], vs[{BlockSize.y}];

                for(unsigned int kz = 0, iz = oz; kz < {KernelDepth}; kz++, iz++){{
                    for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{
                        for(unsigned int kx = 0; kx < {KernelWidth}; kx++){{
                            unsigned int filter_index = (inch + {InChannels} * (outch + {OutChannels} * (kx + {KernelWidth} * (ky + {KernelHeight} * kz)))) * 2;
                    
                            float uv_hi = 0, uv_lo = 0;
                    
                            for(unsigned int ox = ox_offset, ix = ox + kx; ox < ox_offset + {BatchPixels} && ox < outwidth; ox++, ix++){{
                                if(tidx == 0 && outch < {OutChannels}){{
                                    vs[tidy] = outmap[outch + {OutChannels} * (ox + outwidth * (oy + outheight * oz))];
                                }}                
                                if(tidy == 0 && inch < {InChannels}){{
                                    us[tidx] = inmap[inch + {InChannels} * (ix + inwidth * (iy + inheight * iz))];
                                }}
                                __syncthreads();

                                if(inch < {InChannels} && outch < {OutChannels}){{
                                    float u = us[tidx];
                                    float v = vs[tidy];

                                    float uv = u * v;

                                    float tmp = uv_hi;

                                    uv_hi += uv;
                                    uv_lo -= (uv_hi - tmp) - uv;    
                                }}
                                __syncthreads();
                            }}

                            if(inch < {InChannels} && outch < {OutChannels}){{
                                float tmp = atomicAdd(filter + filter_index, uv_hi);
                                atomicAdd(filter + filter_index + 1, uv_lo - (((tmp + uv_hi) - tmp) - uv_hi));
                            }}
                        }}
                    }}
                }}
            }}";

            this.Kernel = new Kernel(code, "kernelproduct_3d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            if(hadd == null) { 
                hadd = new ArrayManipulation.HorizontalAdd();
            }

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

            CudaArray<float> dfloat_filter = 
                CudaArrayReserver<float>.Request(stream, inmap.DeviceID, index:0, InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 2);
            dfloat_filter.ZerosetAsync(stream, InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 2);

            uint mul_per_line = InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * outwidth;

            uint lines_per_execute_mul = MulPerExecute / mul_per_line + 1;
            uint lines_per_execute_pixels = (PixelsPerExecute + outwidth - 1) / outwidth;

            uint lines_per_execute = Math.Min(lines_per_execute_mul, lines_per_execute_pixels);

            uint xsets = (outwidth + BatchPixels - 1) / BatchPixels;

            for (uint th = 0; th < batches; th++) {
                for(uint oz = 0; oz < outdepth; oz++){
                    for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                        uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                        Kernel.Execute(
                            indexes: (InChannels, OutChannels, xsets * lines),
                            block: (BlockSize.x, BlockSize.y, 1),
                            dynamic_shared_memory_bytes: 0,
                            stream,
                            inmap.ElementPtr(th * InChannels * inwidth * inheight * indepth),
                            outmap.ElementPtr(th * OutChannels * outwidth * outheight * outdepth),
                            dfloat_filter,
                            oy_offset, oz,
                            xsets,
                            inwidth, outwidth,
                            inheight, outheight
                        );
                    }
                }
            }
            
            hadd.Execute(stream, dfloat_filter, filter, InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 7) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint inheight) || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[5] is uint indepth) || !Limits.CheckDepth(indepth, KernelDepth)) {
                throw new ArgumentException(nameof(indepth));
            }

            if (!(args[6] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
