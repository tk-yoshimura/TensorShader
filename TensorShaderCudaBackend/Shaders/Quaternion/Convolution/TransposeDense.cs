using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Quaternion.Convolution {

    /// <summary>転置全結合</summary>
    public sealed class TransposeDense : Shader {

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
        public TransposeDense(uint inchannels, uint outchannels, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 4, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels / 4;
            this.OutChannels = outchannels / 4;
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

            __global__ void quaternion_transpose_dense(float4 *inmap, float4 *outmap, float4 *filter) {{

                unsigned int outch = {Defines.IndexX}, th = {Defines.BlockIndexY};
                unsigned int tid = {Defines.ThreadIdX}, threads = {Defines.ThreadsX};

                unsigned int inmap_offset = {InChannels} * th;
                inmap += inmap_offset;

                unsigned int outmap_offset = {OutChannels} * th;
                outmap += outmap_offset;

                __shared__ float4 us[{InChannels}];
                float4 vu_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), vu_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);

                unsigned int filter_idx = outch;

                for(unsigned int inch = tid; inch < {InChannels}; inch += threads){{
                    us[inch] = inmap[inch];
                }}
                __syncthreads();

                if(outch < {OutChannels}){{                        
                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{                            
                        float4 u = us[inch];
                        float4 v = filter[filter_idx];

                        {(GradMode ? "quaternion_mulgrad" : "quaternion_mul")}(vu_hi, vu_lo, v, u);

                        filter_idx += {OutChannels};
                    }}

                    outmap[outch] = ctor_float4(vu_hi.x + vu_lo.x, vu_hi.y + vu_lo.y, vu_hi.z + vu_lo.z, vu_hi.w + vu_lo.w);
                }}
            }}";

            this.Kernel = new Kernel(code, "quaternion_transpose_dense");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint batches = (args[3] as uint?).Value;

            Kernel.Execute(
                indexes: (OutChannels, batches),
                block: (Kernel.DefaultBlockSize(OutChannels), 1),
                dynamic_shared_memory_bytes: 0,
                stream,
                inmap,
                outmap,
                filter
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

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * batches * 4) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * batches * 4) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
