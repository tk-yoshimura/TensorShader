﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Trivector.Convolution {

    /// <summary>1次元逆畳み込み</summary>
    public sealed class Deconvolution1D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public Deconvolution1D(uint inchannels, uint outchannels, uint kwidth, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 3, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(nameof(kwidth));
            }

            this.InChannels = inchannels / 3;
            this.OutChannels = outchannels / 3;
            this.KernelWidth = kwidth;
            this.GradMode = gradmode;

            string code = $@"

            static __inline__ __device__ float3 ctor_float3(float x, float y, float z){{
                float3 t; t.x = x; t.y = y; t.z = z; return t;
            }}

            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo -= (hi - tmp) - val;
            }}

            static __inline__ __device__ void trivector_quaternion_mul(float3 &hi, float3 &lo, float3 v, float4 q){{
                float sx = q.x * q.x, sy = q.y * q.y, sz = q.z * q.z, sw = q.w * q.w; 
                float mx = q.y * q.z, my = q.z * q.w, mz = q.w * q.y; 
                float nx = q.x * q.y, ny = q.x * q.z, nz = q.x * q.w;

                floatfloat_add(hi.x, lo.x, v.x * (sx + sy - sz - sw));
                floatfloat_add(hi.x, lo.x, 2.0 * (v.y * (mx - nz)));
                floatfloat_add(hi.x, lo.x, 2.0 * (v.z * (mz + ny)));

                floatfloat_add(hi.y, lo.y, v.y * (sx - sy + sz - sw));
                floatfloat_add(hi.y, lo.y, 2.0 * (v.z * (my - nx)));
                floatfloat_add(hi.y, lo.y, 2.0 * (v.x * (mx + nz)));

                floatfloat_add(hi.z, lo.z, v.z * (sx - sy - sz + sw));
                floatfloat_add(hi.z, lo.z, 2.0 * (v.x * (mz - ny)));
                floatfloat_add(hi.z, lo.z, 2.0 * (v.y * (my + nx)));
            }}

            static __inline__ __device__ void trivector_quaternion_mulgrad(float3 &hi, float3 &lo, float3 v, float4 q){{
                float sx = q.x * q.x, sy = q.y * q.y, sz = q.z * q.z, sw = q.w * q.w; 
                float mx = q.y * q.z, my = q.z * q.w, mz = q.w * q.y; 
                float nx = q.x * q.y, ny = q.x * q.z, nz = q.x * q.w;

                floatfloat_add(hi.x, lo.x, v.x * (sx + sy - sz - sw));
                floatfloat_add(hi.x, lo.x, 2.0 * (v.y * (mx + nz)));
                floatfloat_add(hi.x, lo.x, 2.0 * (v.z * (mz - ny)));

                floatfloat_add(hi.y, lo.y, v.y * (sx - sy + sz - sw));
                floatfloat_add(hi.y, lo.y, 2.0 * (v.z * (my + nx)));
                floatfloat_add(hi.y, lo.y, 2.0 * (v.x * (mx - nz)));

                floatfloat_add(hi.z, lo.z, v.z * (sx - sy - sz + sw));
                floatfloat_add(hi.z, lo.z, 2.0 * (v.x * (mz + ny)));
                floatfloat_add(hi.z, lo.z, 2.0 * (v.y * (my - nx)));
            }}

            __global__ void trivector_deconvolution_1d(float3 *inmap, float3 *outmap, float4 *filter,
                                                       unsigned int inwidth) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};
                unsigned int ox = {Defines.BlockIndexY};

                __shared__ float3 vs[{InChannels}];
                float3 vq_hi = ctor_float3(0.0, 0.0, 0.0), vq_lo = ctor_float3(0.0, 0.0, 0.0);

                for(unsigned int kx = 0, ix = ox - {KernelWidth - 1}; kx < {KernelWidth}; kx++, ix++){{ 
                    if(ix >= inwidth){{
                        continue;
                    }}

                    unsigned int inmap_idx = {InChannels} * ix;
                    unsigned int filter_idx = outch + {InChannels * OutChannels} * ({KernelWidth - 1} - kx);

                    for(unsigned int inch = tid; inch < {InChannels}; inch += threads){{
                        vs[inch] = inmap[inch + inmap_idx];
                    }}
                    __syncthreads();

                    if(outch < {OutChannels}){{                        
                        for(unsigned int inch = 0; inch < {InChannels}; inch++){{                            
                            float3 v = vs[inch];
                            float4 q = filter[filter_idx];

                            {(GradMode ? "trivector_quaternion_mulgrad" : "trivector_quaternion_mul")}(vq_hi, vq_lo, v, q);

                            filter_idx += {OutChannels};
                        }}

                    }}
                    __syncthreads();
                }}

                if(outch < {OutChannels}){{
                    unsigned int outmap_idx = outch + {OutChannels} * ox;

                    outmap[outmap_idx] = ctor_float3(vq_hi.x + vq_lo.x, vq_hi.y + vq_lo.y, vq_hi.z + vq_lo.z);
                }}
            }}";

            this.Kernel = new Kernel(code, "trivector_deconvolution_1d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            uint outwidth = inwidth + KernelWidth - 1;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (OutChannels, outwidth),
                    block: (Kernel.DefaultBlockSize(OutChannels), 1),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * InChannels * inwidth * 3),
                    outmap.ElementPtr(th * OutChannels * outwidth * 3),
                    filter,
                    inwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + KernelWidth - 1;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * batches * 3) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * batches * 3) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}