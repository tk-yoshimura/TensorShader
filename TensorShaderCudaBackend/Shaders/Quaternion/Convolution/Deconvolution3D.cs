﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Quaternion.Convolution {

    /// <summary>3次元逆畳み込み</summary>
    public sealed class Deconvolution3D : Shader {

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

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>実行あたりの積数(2^25=33554432‬)</summary>
        public static uint MulPerExecute => 0x2000000;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(KernelDepth)} = {KernelDepth} " +
            $"{nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public Deconvolution3D(uint inchannels, uint outchannels, uint kwidth, uint kheight, uint kdepth, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 4, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight, kdepth)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}, {nameof(kdepth)}");
            }

            this.InChannels = inchannels / 4;
            this.OutChannels = outchannels / 4;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.GradMode = gradmode;

            string code = $@"

            static __inline__ __device__ float4 ctor_float4(float x, float y, float z, float w){{
                float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
            }}

            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo -= (hi - tmp) - val;
            }}

            static __inline__ __device__ void floatfloat_sub(float &hi, float &lo, float val){{
                float tmp = hi;
                hi -= val;
                lo -= (hi - tmp) + val;
            }}

            static __inline__ __device__ void quaternion_mul(float4 &hi, float4 &lo, float4 x1, float4 x2){{
                floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                floatfloat_sub(hi.x, lo.x, x1.y * x2.y);
                floatfloat_sub(hi.x, lo.x, x1.z * x2.z);
                floatfloat_sub(hi.x, lo.x, x1.w * x2.w);

                floatfloat_add(hi.y, lo.y, x1.x * x2.y);
                floatfloat_add(hi.y, lo.y, x1.y * x2.x);
                floatfloat_sub(hi.y, lo.y, x1.z * x2.w);
                floatfloat_add(hi.y, lo.y, x1.w * x2.z);

                floatfloat_add(hi.z, lo.z, x1.x * x2.z);
                floatfloat_add(hi.z, lo.z, x1.y * x2.w);
                floatfloat_add(hi.z, lo.z, x1.z * x2.x);
                floatfloat_sub(hi.z, lo.z, x1.w * x2.y);

                floatfloat_add(hi.w, lo.w, x1.x * x2.w);
                floatfloat_sub(hi.w, lo.w, x1.y * x2.z);
                floatfloat_add(hi.w, lo.w, x1.z * x2.y);
                floatfloat_add(hi.w, lo.w, x1.w * x2.x);
            }}

            static __inline__ __device__ void quaternion_mulgrad(float4 &hi, float4 &lo, float4 x1, float4 x2){{
                floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                floatfloat_add(hi.x, lo.x, x1.y * x2.y);
                floatfloat_add(hi.x, lo.x, x1.z * x2.z);
                floatfloat_add(hi.x, lo.x, x1.w * x2.w);

                floatfloat_add(hi.y, lo.y, x1.x * x2.y);
                floatfloat_sub(hi.y, lo.y, x1.y * x2.x);
                floatfloat_add(hi.y, lo.y, x1.z * x2.w);
                floatfloat_sub(hi.y, lo.y, x1.w * x2.z);

                floatfloat_add(hi.z, lo.z, x1.x * x2.z);
                floatfloat_sub(hi.z, lo.z, x1.y * x2.w);
                floatfloat_sub(hi.z, lo.z, x1.z * x2.x);
                floatfloat_add(hi.z, lo.z, x1.w * x2.y);

                floatfloat_add(hi.w, lo.w, x1.x * x2.w);
                floatfloat_add(hi.w, lo.w, x1.y * x2.z);
                floatfloat_sub(hi.w, lo.w, x1.z * x2.y);
                floatfloat_sub(hi.w, lo.w, x1.w * x2.x);
            }}

            __global__ void quaternion_deconvolution_3d(float4 *inmap, float4 *outmap, float4 *filter,
                                                        unsigned int oy_offset, unsigned int oz, 
                                                        unsigned int inwidth, unsigned int outwidth, 
                                                        unsigned int inheight, unsigned int outheight, 
                                                        unsigned int indepth) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int ox = {Defines.BlockIndexY}, oy = oy_offset + {Defines.BlockIndexZ};

                __shared__ float4 us[{InChannels}];
                float4 vu_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), vu_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);
            
                for(unsigned int kz = 0, iz = oz - {KernelDepth - 1}; kz < {KernelDepth}; kz++, iz++){{ 
                    if(iz >= indepth){{
                        continue;
                    }}

                    for(unsigned int ky = 0, iy = oy - {KernelHeight - 1}; ky < {KernelHeight}; ky++, iy++){{ 
                        if(iy >= inheight){{
                            continue;
                        }}

                        for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{ 
                            if(ix >= inwidth){{
                                continue;
                            }}

                            unsigned int inmap_idx = {InChannels} * (ix + inwidth * (iy + inheight * iz));
                            unsigned int filter_idx = outch + {InChannels * OutChannels} * 
                                                      (({KernelWidth - 1} - kx) + {KernelWidth} * (({KernelHeight - 1} - ky) + {KernelHeight} * ({KernelDepth - 1} - kz)));

                            for(unsigned int inch = tid; inch < {InChannels}; inch += threads){{
                                us[inch] = inmap[inch + inmap_idx];
                            }}
                            __syncthreads();

                            if(outch < {OutChannels}){{                        
                                for(unsigned int inch = 0; inch < {InChannels}; inch++){{                            
                                    float4 u = us[inch];
                                    float4 v = filter[filter_idx];

                                    {(GradMode ? "quaternion_mulgrad" : "quaternion_mul")}(vu_hi, vu_lo, v, u);

                                    filter_idx += {OutChannels};
                                }}

                            }}
                            __syncthreads();
                        }}
                    }}
                }}

                if(outch < {OutChannels}){{
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * (oy + outheight * oz));

                    outmap[outmap_idx] = ctor_float4(vu_hi.x + vu_lo.x, vu_hi.y + vu_lo.y, vu_hi.z + vu_lo.z, vu_hi.w + vu_lo.w);
                }}
            }}";

            this.Kernel = new Kernel(code, "quaternion_deconvolution_3d");
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

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;
            uint outdepth = indepth + KernelDepth - 1;

            uint mul_per_line = InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * outwidth * 16;

            uint lines_per_execute = MulPerExecute / mul_per_line + 1;

            for (uint th = 0; th < batches; th++) {
                for (uint oz = 0; oz < outdepth; oz++) {
                    for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                        uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                        Kernel.Execute(
                            indexes: (OutChannels, outwidth, lines),
                            block: (Kernel.DefaultBlockSize(OutChannels), 1, 1),
                            dynamic_shared_memory_bytes: 0,
                            stream,
                            inmap.ElementPtr(th * InChannels * inwidth * inheight * indepth * 4),
                            outmap.ElementPtr(th * OutChannels * outwidth * outheight * outdepth * 4),
                            filter,
                            oy_offset, oz,
                            inwidth, outwidth, inheight, outheight, indepth
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

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint inheight) || !Limits.CheckHeight(inheight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[5] is uint indepth) || !Limits.CheckDepth(indepth)) {
                throw new ArgumentException(nameof(indepth));
            }

            if (!(args[6] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + KernelWidth - 1;
            uint outheight = inheight + KernelHeight - 1;
            uint outdepth = indepth + KernelDepth - 1;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * inheight * indepth * batches * 4) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * outheight * outdepth * batches * 4) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight * KernelDepth * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}