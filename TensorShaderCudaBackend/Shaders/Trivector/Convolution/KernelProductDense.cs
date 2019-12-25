using System;
using System.Linq;

using static TensorShaderCudaBackend.Elementwise;
using static TensorShaderCudaBackend.ArrayManipulation;

namespace TensorShaderCudaBackend.Shaders.Trivector.Convolution {

    /// <summary>カーネル積</summary>
    public sealed class KernelProductDense : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>ブロックサイズ</summary>
        public (uint x, uint y) BlockSize { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => 
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " + 
            $"{nameof(Transpose)} = {Transpose}";
        
        /// <summary>コンストラクタ</summary>
        public KernelProductDense(uint inchannels, uint outchannels, bool transpose) { 
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple:3, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels / 3;
            this.OutChannels = outchannels / 3;
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

            __global__ void trivector_kernelproduct_dense(float3 *inmap, float3 *outmap, float4 *filter_value, float4 *filter_grad) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY}, th = {Defines.BlockIndexZ};
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                unsigned int inmap_offset = {InChannels} * th;
                inmap += inmap_offset;

                unsigned int outmap_offset = {OutChannels} * th;
                outmap += outmap_offset;
                
                unsigned int filter_offset = inch + {InChannels} * outch;
                filter_value += filter_offset;
                filter_grad += filter_offset * 2;

                __shared__ float3 us[{BlockSize.x}], vs[{BlockSize.y}];

                if(tidx == 0 && outch < {OutChannels}){{
                    vs[tidy] = outmap[outch];
                }}                
                if(tidy == 0 && inch < {InChannels}){{
                    us[tidx] = inmap[inch];
                }}
                __syncthreads();

                if(inch < {InChannels} && outch < {OutChannels}){{
                    float3 u = us[tidx];
                    float3 v = vs[tidy];
                    float4 q = filter_value[0];

                    float4 gq_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), gq_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);

                    trivector_quaternion_kernelprod(gq_hi, gq_lo, {(Transpose ? "v, u" : "u, v")}, q);

                    floatfloat_atomicadd(filter_grad, gq_hi, gq_lo);
                }}
            }}";

            this.Kernel = new Kernel(code, "trivector_kernelproduct_dense");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter_value = args[2] as CudaArray<float>;
            CudaArray<float> filter_grad = args[3] as CudaArray<float>;
           
            uint batches = (args[4] as uint?).Value;

            CudaArray<float> dfloat_filter = 
                CudaArrayReserver<float>.Request(stream, inmap.DeviceID, index:0, InChannels * OutChannels * 8);
            dfloat_filter.ZerosetAsync(stream, InChannels * OutChannels * 8);

            Kernel.Execute(
                indexes:(InChannels, OutChannels, batches), 
                block:(BlockSize.x, BlockSize.y, 1),
                dynamic_shared_memory_bytes: 0, 
                stream,
                inmap, 
                outmap,
                filter_value,
                dfloat_filter
            );
            
            HorizontalAdd(InChannels * OutChannels * 4, dfloat_filter, filter_grad, stream);
            MulConstant(InChannels * OutChannels * 4, 2, filter_grad, filter_grad, stream);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }
            
            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * batches * 3) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * batches * 3) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter_value) || filter_value.Length < InChannels * OutChannels * 4) {
                throw new ArgumentException(nameof(filter_value));
            }

            if (!(args[3] is CudaArray<float> filter_grad) || filter_grad.Length < InChannels * OutChannels * 4) {
                throw new ArgumentException(nameof(filter_grad));
            }
        }
    }
}
