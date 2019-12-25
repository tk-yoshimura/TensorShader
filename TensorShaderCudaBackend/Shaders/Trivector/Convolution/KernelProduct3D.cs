using System;
using System.Linq;

using static TensorShaderCudaBackend.Elementwise;
using static TensorShaderCudaBackend.ArrayManipulation;

namespace TensorShaderCudaBackend.Shaders.Trivector.Convolution {

    /// <summary>カーネル積</summary>
    public sealed class KernelProduct3D : Shader {

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

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>実行あたりの積数(2^24=16777216‬)</summary>
        public static uint MulPerExecute => 0x1000000;

        /// <summary>実行あたりのピクセル数(2^14=16384‬)</summary>
        public static uint PixelsPerExecute => 0x4000;

        /// <summary>ブロックサイズ</summary>
        public (uint x, uint y) BlockSize { private set; get; }

        /// <summary>1スレッドで処理する対象ピクセル数</summary>
        private static uint BatchPixels => 16;
        
        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " + 
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(KernelDepth)} = {KernelDepth} " +
            $"{nameof(Transpose)} = {Transpose}";
        
        /// <summary>コンストラクタ</summary>
        public KernelProduct3D(uint inchannels, uint outchannels, uint kwidth, uint kheight, uint kdepth, bool transpose) { 
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple:3, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight, kdepth)) { 
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}, {nameof(kdepth)}");
            }

            this.InChannels = inchannels / 3;
            this.OutChannels = outchannels / 3;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.Transpose = transpose;

            this.BlockSize = Kernel.MinimizeGridsBlockSize((InChannels, OutChannels));

            string code = $@"

            static __inline__ __device__ float4 ctor_float4(float x, float y, float z, float w){{
                float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
            }}

            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo -= (hi - tmp) - val;
            }}

            static __inline__ __device__ void trivector_quaternion_kernelprod(float4 &hi, float4 &lo, float3 v, float3 u, float4 q){{
                float vxqx = v.x * q.x, vxqy = v.x * q.y, vxqz = v.x * q.z, vxqw = v.x * q.w;
                float vyqx = v.y * q.x, vyqy = v.y * q.y, vyqz = v.y * q.z, vyqw = v.y * q.w;
                float vzqx = v.z * q.x, vzqy = v.z * q.y, vzqz = v.z * q.z, vzqw = v.z * q.w;

                floatfloat_add(hi.x, lo.x, u.x * (vzqz + vxqx - vyqw));
                floatfloat_add(hi.x, lo.x, u.y * (vxqw + vyqx - vzqy));
                floatfloat_add(hi.x, lo.x, u.z * (vyqy + vzqx - vxqz));

                floatfloat_add(hi.y, lo.y, u.x * (vzqw + vxqy + vyqz));
                floatfloat_add(hi.y, lo.y, u.y * (vxqz - vyqy - vzqx));
                floatfloat_add(hi.y, lo.y, u.z * (vyqx - vzqy + vxqw));

                floatfloat_add(hi.z, lo.z, u.x * (vzqx - vxqz + vyqy));
                floatfloat_add(hi.z, lo.z, u.y * (vxqy + vyqz + vzqw));
                floatfloat_add(hi.z, lo.z, u.z * (vyqw - vzqz - vxqx));

                floatfloat_add(hi.w, lo.w, u.x * (vzqy - vxqw - vyqx));
                floatfloat_add(hi.w, lo.w, u.y * (vxqx - vyqw + vzqz));
                floatfloat_add(hi.w, lo.w, u.z * (vyqz + vzqw + vxqy));
            }}

            static __inline__ __device__ void floatfloat_atomicadd(float4 *ptr, float4 hi, float4 lo){{
                float *ptr_float = (float*)(void*)ptr;

                float tmpx = atomicAdd(ptr_float, hi.x);
                atomicAdd(ptr_float + 1, lo.x - (((tmpx + hi.x) - tmpx) - hi.x));
                float tmpy = atomicAdd(ptr_float + 2, hi.y);
                atomicAdd(ptr_float + 3, lo.y - (((tmpy + hi.y) - tmpy) - hi.y));
                float tmpz = atomicAdd(ptr_float + 4, hi.z);
                atomicAdd(ptr_float + 5, lo.z - (((tmpz + hi.z) - tmpz) - hi.z));
                float tmpw = atomicAdd(ptr_float + 6, hi.w);
                atomicAdd(ptr_float + 7, lo.w - (((tmpw + hi.w) - tmpw) - hi.w));
            }}

            __global__ void trivector_kernelproduct_3d(float3 *inmap, float3 *outmap, float4 *filter_value, float4 *filter_grad, 
                                                       unsigned int oy_offset, 
                                                       unsigned int oz,
                                                       unsigned int xsets,
                                                       unsigned int inwidth, unsigned int outwidth, 
                                                       unsigned int inheight, unsigned int outheight) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY};
                unsigned int ox_offset = ({Defines.BlockIndexZ} % xsets) * {BatchPixels};
                unsigned int oy = oy_offset + {Defines.BlockIndexZ} / xsets;
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                __shared__ float3 us[{BlockSize.x}], vs[{BlockSize.y}];

                for(unsigned int kz = 0, iz = oz; kz < {KernelDepth}; kz++, iz++){{
                    for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{
                        for(unsigned int kx = 0; kx < {KernelWidth}; kx++){{
                            unsigned int filter_index = (inch + {InChannels} * (outch + {OutChannels} * (kx + {KernelWidth} * (ky + {KernelHeight} * kz))));

                            float4 q = filter_value[filter_index];
                    
                            float4 gq_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), gq_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);
                    
                            for(unsigned int ox = ox_offset, ix = ox + kx; ox < ox_offset + {BatchPixels} && ox < outwidth; ox++, ix++){{
                                if(tidx == 0 && outch < {OutChannels}){{
                                    vs[tidy] = outmap[outch + {OutChannels} * (ox + outwidth * (oy + outheight * oz))];
                                }}                
                                if(tidy == 0 && inch < {InChannels}){{
                                    us[tidx] = inmap[inch + {InChannels} * (ix + inwidth * (iy + inheight * iz))];
                                }}
                                __syncthreads();

                                if(inch < {InChannels} && outch < {OutChannels}){{
                                    float3 u = us[tidx];
                                    float3 v = vs[tidy];

                                    trivector_quaternion_kernelprod(gq_hi, gq_lo, {(Transpose ? "v, u" : "u, v")}, q);
                                }}
                                __syncthreads();
                            }}

                            if(inch < {InChannels} && outch < {OutChannels}){{
                                floatfloat_atomicadd(filter_grad + filter_index * 2, gq_hi, gq_lo);
                            }}
                        }}
                    }}
                }}
            }}";

            this.Kernel = new Kernel(code, "trivector_kernelproduct_3d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter_value = args[2] as CudaArray<float>;
            CudaArray<float> filter_grad = args[3] as CudaArray<float>;
           
            uint inwidth = (args[4] as uint?).Value;
            uint inheight = (args[5] as uint?).Value;
            uint indepth = (args[6] as uint?).Value;
            uint batches = (args[7] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            CudaArray<float> dfloat_filter = 
                CudaArrayReserver<float>.Request(stream, inmap.DeviceID, index:0, InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 8);
            dfloat_filter.ZerosetAsync(stream, InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 8);

            uint mul_per_line = InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * outwidth * 16;

            uint lines_per_execute_mul = MulPerExecute / mul_per_line + 1;
            uint lines_per_execute_pixels = (PixelsPerExecute + outwidth - 1) / outwidth;

            uint lines_per_execute = Math.Min(lines_per_execute_mul, lines_per_execute_pixels);

            uint xsets = (outwidth + BatchPixels - 1) / BatchPixels;

            for (uint th = 0; th < batches; th++) {
                for (uint oz = 0; oz < outdepth; oz++) {
                    for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                        uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                        Kernel.Execute(
                            indexes: (InChannels, OutChannels, xsets * lines),
                            block: (BlockSize.x, BlockSize.y, 1),
                            dynamic_shared_memory_bytes: 0,
                            stream,
                            inmap.ElementPtr(th * InChannels * inwidth * inheight * indepth * 3),
                            outmap.ElementPtr(th * OutChannels * outwidth * outheight * outdepth * 3),
                            filter_value,
                            dfloat_filter,
                            oy_offset, oz,
                            xsets,
                            inwidth, outwidth, inheight, outheight
                        );
                    }
                }
            }
            
            HorizontalAdd(InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 4, dfloat_filter, filter_grad, stream);
            MulConstant(InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 4, 2, filter_grad, filter_grad, stream);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 8) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[4] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[5] is uint inheight) || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[6] is uint indepth) || !Limits.CheckDepth(indepth, KernelDepth)) {
                throw new ArgumentException(nameof(indepth));
            }

            if (!(args[7] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;
            uint outdepth = indepth + 1 - KernelDepth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * inheight * indepth * batches * 3) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * outheight * outdepth * batches * 3) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter_value) || filter_value.Length < InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 4) {
                throw new ArgumentException(nameof(filter_value));
            }

            if (!(args[3] is CudaArray<float> filter_grad) || filter_grad.Length < InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 4) {
                throw new ArgumentException(nameof(filter_grad));
            }
        }
    }
}
