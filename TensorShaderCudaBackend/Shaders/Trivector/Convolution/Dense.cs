﻿using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Trivector.Convolution {

    /// <summary>全結合</summary>
    public sealed class Dense : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public Dense(uint inchannels, uint outchannels, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 3, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels / 3;
            this.OutChannels = outchannels / 3;
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

            __global__ void trivector_dense(float3 *inmap, float3 *outmap, float4 *filter) {{

                unsigned int outch = {Defines.IndexX}, th = {Defines.BlockIndexY};
                unsigned int tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};

                unsigned int inmap_offset = {InChannels} * th;
                inmap += inmap_offset;

                unsigned int outmap_offset = {OutChannels} * th;
                outmap += outmap_offset;

                __shared__ float3 vs[{InChannels}];
                float3 vq_hi = ctor_float3(0.0, 0.0, 0.0), vq_lo = ctor_float3(0.0, 0.0, 0.0);

                unsigned int filter_idx = outch;

                for(unsigned int inch = tid; inch < {InChannels}; inch += threads){{
                    vs[inch] = inmap[inch];
                }}
                __syncthreads();

                if(outch < {OutChannels}){{                        
                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{                            
                        float3 v = vs[inch];
                        float4 q = filter[filter_idx];

                        {(GradMode ? "trivector_quaternion_mulgrad" : "trivector_quaternion_mul")}(vq_hi, vq_lo, v, q);

                        filter_idx += {OutChannels};
                    }}

                    outmap[outch] = ctor_float3(vq_hi.x + vq_lo.x, vq_hi.y + vq_lo.y, vq_hi.z + vq_lo.z);
                }}
            }}";

            this.Kernel = new Kernel(code, "trivector_dense");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint batches = (args[3] as uint?).Value;

            CudaArray<float> transpose_filter =
                CudaArrayReserver<float>.Request(stream, filter.DeviceID, index: 0, InChannels * OutChannels * 4);

            TransposeQuaternionKernelChannel(InChannels * 4, OutChannels * 4, 1u, filter, transpose_filter, stream);

            Kernel.Execute(
                indexes: (OutChannels, batches),
                block: (Kernel.DefaultBlockSize(OutChannels), 1),
                dynamic_shared_memory_bytes: 0, stream,
                inmap,
                outmap,
                transpose_filter
            );
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * batches * 3) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * batches * 3) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}