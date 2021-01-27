using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>2次元畳み込み</summary>
    public sealed class Convolution2D_ConstChXY : Convolution2D_Base {

        /// <summary>コンストラクタ</summary>
        public Convolution2D_ConstChXY(uint inchannels, uint outchannels, uint kwidth, uint kheight) 
            :base(inchannels, outchannels, kwidth, kheight) {

            string code = $@"

            {Defines.FloatFloatFma}
            {Defines.StoreFloatSharedMemory(elemsize: 1, InChannels, ThreadsX)}

            __constant__ float filter[{InChannels * OutChannels * KernelWidth * KernelHeight}];

            __global__ void convolution_2d_constcxy(const float* __restrict__ inmap, float* __restrict__ outmap,
                                                    unsigned int oy_offset,
                                                    unsigned int inwidth, unsigned int outwidth) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                __shared__ float us[{InChannels}];
                float uv_hi = 0.0, uv_lo = 0.0;

                unsigned int filter_idx = outch;

                for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{

                    unsigned int inmap_idx = {InChannels} * (ox + inwidth * iy);

                    { (KernelWidth <= 7 ? "#pragma unroll" : "") }
                    for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{

                        store_smem(inmap + inmap_idx, us, tid);
                        inmap_idx += {InChannels};

                        { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                            #pragma unroll 8
                            for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                                float u = us[inch];
                                float v = filter[filter_idx];

                                floatfloat_fma(uv_hi, uv_lo, u, v);

                                filter_idx += {OutChannels};
                            }}

                        { (OutChannels % ThreadsX != 0 ? "}" : "") }
                        __syncthreads();
                    }}
                }}

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * oy);

                    outmap[outmap_idx] = uv_hi + uv_lo;
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "convolution_2d_constcxy");
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

            ulong mul_per_line = (ulong)InChannels * OutChannels * KernelWidth * KernelHeight * outwidth;

            uint lines_per_execute = (uint)(MulPerExecute / mul_per_line + 1);

            CudaArray<float> transpose_filter =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 0, InChannels * OutChannels * KernelWidth * KernelHeight);

            TransposeKernelChannel(InChannels, OutChannels, KernelWidth * KernelHeight, filter, transpose_filter, stream);

            if (stream != null) {
                Kernel.StoreConstMemoryAsync(stream, "filter", transpose_filter, InChannels * OutChannels * KernelWidth * KernelHeight);
            }
            else {
                Kernel.StoreConstMemory("filter", transpose_filter, InChannels * OutChannels * KernelWidth * KernelHeight);
            }

            for (uint th = 0; th < batches; th++) {
                for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                    uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                    Kernel.Execute(
                        indexes: (OutChannels, outwidth, lines),
                        block: (ThreadsX, 1, 1),
                        dynamic_shared_memory_bytes: 0, stream,
                        inmap.ElementPtr(th * InChannels * inwidth * inheight),
                        outmap.ElementPtr(th * OutChannels * outwidth * outheight),
                        oy_offset,
                        inwidth, outwidth
                    );
                }
            }
        }
    }
}
