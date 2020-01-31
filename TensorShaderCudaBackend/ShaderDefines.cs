using System.Linq;

namespace TensorShaderCudaBackend {

    /// <summary>コンピュートシェーダー</summary>
    public abstract partial class Shader {

        /// <summary>定義済み変数/関数</summary>
        protected static class Defines {
            /// <summary>Xインデクス</summary>
            public static string IndexX => "(blockDim.x * blockIdx.x + threadIdx.x)";

            /// <summary>Yインデクス</summary>
            public static string IndexY => "(blockDim.y * blockIdx.y + threadIdx.y)";

            /// <summary>Zインデクス</summary>
            public static string IndexZ => "(blockDim.z * blockIdx.z + threadIdx.z)";

            /// <summary>Xブロックインデクス</summary>
            public static string BlockIndexX => "(blockIdx.x)";

            /// <summary>Yブロックインデクス</summary>
            public static string BlockIndexY => "(blockIdx.y)";

            /// <summary>Zブロックインデクス</summary>
            public static string BlockIndexZ => "(blockIdx.z)";

            /// <summary>XスレッドID</summary>
            public static string ThreadIdX => "(threadIdx.x)";

            /// <summary>YスレッドID</summary>
            public static string ThreadIdY => "(threadIdx.y)";

            /// <summary>Xスレッド数</summary>
            public static string ThreadsX => "(blockDim.x)";

            /// <summary>Yスレッド数</summary>
            public static string ThreadsY => "(blockDim.y)";

            /// <summary>FloatFloat加算</summary>
            public static string FloatFloatAdd =>
            $@"
            static __inline__ __device__ void floatfloat_add(float &hi, float &lo, float val){{
                float tmp = hi;
                hi += val;
                lo += val - (hi - tmp);
            }}";

            /// <summary>FloatFloat減算</summary>
            public static string FloatFloatSub =>
            $@"
            static __inline__ __device__ void floatfloat_sub(float &hi, float &lo, float val){{
                float tmp = hi;
                hi -= val;
                lo -= val + (hi - tmp);
            }}";

            /// <summary>FloatFloat融合積和演算</summary>
            public static string FloatFloatFma =>
            $@"
            static __inline__ __device__ void floatfloat_fma(float &hi, float &lo, float val_x, float val_y){{
                float tmp = hi;
                hi = fmaf(val_x, val_y, hi);
                lo += fmaf(val_x, val_y, tmp - hi);
            }}";

            /// <summary>FloatFloat融合積差演算</summary>
            public static string FloatFloatFms =>
            $@"
            static __inline__ __device__ void floatfloat_fms(float &hi, float &lo, float val_x, float val_y){{
                float tmp = hi;
                hi = fmaf(-val_x, val_y, hi);
                lo -= fmaf(val_x, val_y, hi - tmp);
            }}";

            /// <summary>FloatFloat加算</summary>
            public static string FloatFloatHiLoAdd =>
            $@"
            static __inline__ __device__ void floatfloat_hilo_add(float &hi, float &lo, float val_hi, float val_lo){{
                float tmp = hi;
                hi += val_hi;
                lo += val_lo + (val_hi - (hi - tmp));
            }}";

            /// <summary>Float2コンストラクタ</summary>
            public static string CtorFloat2 =>
            $@"
            static __inline__ __device__ float2 ctor_float2(float x, float y){{
                float2 t; t.x = x; t.y = y; return t;
            }}";

            /// <summary>Float3コンストラクタ</summary>
            public static string CtorFloat3 =>
            $@"
            static __inline__ __device__ float3 ctor_float3(float x, float y, float z){{
                float3 t; t.x = x; t.y = y; t.z = z; return t;
            }}";

            /// <summary>Float4コンストラクタ</summary>
            public static string CtorFloat4 =>
            $@"
            static __inline__ __device__ float4 ctor_float4(float x, float y, float z, float w){{
                float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
            }}";

            /// <summary>原子性保証加算</summary>
            public static string AtomicAdd =>
            $@"
            static __inline__ __device__ void floatfloat_atomicadd(float *ptr, float hi, float lo){{
                float tmp = atomicAdd(ptr, hi);
                atomicAdd(ptr + 1, lo - (((tmp + hi) - tmp) - hi));
            }}";

            /// <summary>シェアードメモリへ格納</summary>
            public static string StoreFloatSharedMemory(uint elemsize, uint elements, uint threads){
                string elem = elemsize > 1 ? $"float{elemsize}" : "float";
                uint length = elements * elemsize;

                string declare = $"static __inline__ __device__ void store_smem(const {elem}* __restrict__ ptr, {elem} *smem, unsigned int thread_idx)";

                string repointer = @"
                        const float* __restrict__ ptr_const = (const float* __restrict__)ptr;
                        volatile float* smem_volatile = (volatile float*)smem;
                    ";
                
                if(threads > length){
                    return $@"
                    {declare}{{
                        {repointer}
                        if(thread_idx < {length}) smem_volatile[thread_idx] = ptr_const[thread_idx];
                    }}";
                } 
                else if(threads == length){ 
                    return $@"
                    {declare}{{
                        {repointer}
                        smem_volatile[thread_idx] = ptr_const[thread_idx];
                    }}";
                }
                else if(threads * 8 >= length && length % threads == 0){
                    return $@"
                    {declare}{{
                        {repointer}
                        unsigned int i = thread_idx;
                        { string.Join(" ", Enumerable.Repeat($"smem_volatile[i] = ptr_const[i]; i += {threads};", (int)(length / threads))) }
                    }}";
                }
                else if(threads * 8 >= length){ 
                    return $@"
                    {declare}{{
                        {repointer}
                        unsigned int i = thread_idx;
                        { string.Join(" ", Enumerable.Repeat($"smem_volatile[i] = ptr_const[i]; i += {threads};", (int)(length / threads))) }
                        if(i < {length}) smem_volatile[i] = ptr_const[i];
                    }}";
                }
                else{
                    return $@"
                    {declare}{{ 
                        {repointer}
                        #pragma unroll 8
                        for(unsigned int i = thread_idx; i < {length}; i += {threads}){{
                            smem_volatile[i] = ptr_const[i];
                        }}
                    }}";
                }
            }

            /// <summary>複素数</summary>
            public static class Complex {
                /// <summary>カーネル積</summary>
                public static string KernelProd =>
                $@"
                static __inline__ __device__ void complex_kernelprod(float2 &hi, float2 &lo, float2 x1, float2 x2){{
                    float val_hi, val_lo;

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.x);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.y);
                    floatfloat_hilo_add(hi.x, lo.x, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fms(val_hi, val_lo, x1.y, x2.x);
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.y);
                    floatfloat_hilo_add(hi.y, lo.y, val_hi, val_lo);
                }}";

                /// <summary>積</summary>
                public static string Mul =>
                $@"
                static __inline__ __device__ void complex_mul(float2 &hi, float2 &lo, float2 x1, float2 x2){{
                    float val_hi, val_lo;
                    
                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.x);
                    floatfloat_fms(val_hi, val_lo, x1.y, x2.y);
                    floatfloat_hilo_add(hi.x, lo.x, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.y);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.x);
                    floatfloat_hilo_add(hi.y, lo.y, val_hi, val_lo);
                }}";

                /// <summary>積勾配</summary>
                public static string MulGrad =>
                $@"
                static __inline__ __device__ void complex_mulgrad(float2 &hi, float2 &lo, float2 x1, float2 x2){{
                    float val_hi, val_lo;

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.x);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.y);
                    floatfloat_hilo_add(hi.x, lo.x, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.x);
                    floatfloat_fms(val_hi, val_lo, x1.x, x2.y);
                    floatfloat_hilo_add(hi.y, lo.y, val_hi, val_lo);
                }}";

                /// <summary>原子性保証加算</summary>
                public static string AtomicAdd =>
                $@"
                static __inline__ __device__ void floatfloat_atomicadd(float2 *ptr, float2 hi, float2 lo){{
                    float *ptr_float = (float*)ptr;

                    float tmpx = atomicAdd(ptr_float, hi.x);
                    atomicAdd(ptr_float + 1, lo.x - (((tmpx + hi.x) - tmpx) - hi.x));
                    float tmpy = atomicAdd(ptr_float + 2, hi.y);
                    atomicAdd(ptr_float + 3, lo.y - (((tmpy + hi.y) - tmpy) - hi.y));
                }}";
            }

            /// <summary>四元数</summary>
            public static class Quaternion {
                /// <summary>カーネル積</summary>
                public static string KernelProd =>
                $@"
                static __inline__ __device__ void quaternion_kernelprod(float4 &hi, float4 &lo, float4 x1, float4 x2){{
                    float val_hi, val_lo;

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.x);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.y);
                    floatfloat_fma(val_hi, val_lo, x1.z, x2.z);
                    floatfloat_fma(val_hi, val_lo, x1.w, x2.w);
                    floatfloat_hilo_add(hi.x, lo.x, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.y);
                    floatfloat_fms(val_hi, val_lo, x1.y, x2.x);
                    floatfloat_fms(val_hi, val_lo, x1.z, x2.w);
                    floatfloat_fma(val_hi, val_lo, x1.w, x2.z);
                    floatfloat_hilo_add(hi.y, lo.y, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.z);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.w);
                    floatfloat_fms(val_hi, val_lo, x1.z, x2.x);
                    floatfloat_fms(val_hi, val_lo, x1.w, x2.y);
                    floatfloat_hilo_add(hi.z, lo.z, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.w);
                    floatfloat_fms(val_hi, val_lo, x1.y, x2.z);
                    floatfloat_fma(val_hi, val_lo, x1.z, x2.y);
                    floatfloat_fms(val_hi, val_lo, x1.w, x2.x);
                    floatfloat_hilo_add(hi.w, lo.w, val_hi, val_lo);
                }}";

                /// <summary>積</summary>
                public static string Mul =>
                $@"
                static __inline__ __device__ void quaternion_mul(float4 &hi, float4 &lo, float4 x1, float4 x2){{
                    float val_hi, val_lo;

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.x);
                    floatfloat_fms(val_hi, val_lo, x1.y, x2.y);
                    floatfloat_fms(val_hi, val_lo, x1.z, x2.z);
                    floatfloat_fms(val_hi, val_lo, x1.w, x2.w);
                    floatfloat_hilo_add(hi.x, lo.x, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.y);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.x);
                    floatfloat_fma(val_hi, val_lo, x1.z, x2.w);
                    floatfloat_fms(val_hi, val_lo, x1.w, x2.z);
                    floatfloat_hilo_add(hi.y, lo.y, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.z);
                    floatfloat_fms(val_hi, val_lo, x1.y, x2.w);
                    floatfloat_fma(val_hi, val_lo, x1.z, x2.x);
                    floatfloat_fma(val_hi, val_lo, x1.w, x2.y);
                    floatfloat_hilo_add(hi.z, lo.z, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.w);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.z);
                    floatfloat_fms(val_hi, val_lo, x1.z, x2.y);
                    floatfloat_fma(val_hi, val_lo, x1.w, x2.x);
                    floatfloat_hilo_add(hi.w, lo.w, val_hi, val_lo);
                }}";

                /// <summary>積勾配</summary>
                public static string MulGrad =>
                $@"
                static __inline__ __device__ void quaternion_mulgrad(float4 &hi, float4 &lo, float4 x1, float4 x2){{
                    float val_hi, val_lo;

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, x1.x, x2.x);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.y);
                    floatfloat_fma(val_hi, val_lo, x1.z, x2.z);
                    floatfloat_fma(val_hi, val_lo, x1.w, x2.w);
                    floatfloat_hilo_add(hi.x, lo.x, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fms(val_hi, val_lo, x1.x, x2.y);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.x);
                    floatfloat_fms(val_hi, val_lo, x1.z, x2.w);
                    floatfloat_fma(val_hi, val_lo, x1.w, x2.z);
                    floatfloat_hilo_add(hi.y, lo.y, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fms(val_hi, val_lo, x1.x, x2.z);
                    floatfloat_fma(val_hi, val_lo, x1.y, x2.w);
                    floatfloat_fma(val_hi, val_lo, x1.z, x2.x);
                    floatfloat_fms(val_hi, val_lo, x1.w, x2.y);
                    floatfloat_hilo_add(hi.z, lo.z, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fms(val_hi, val_lo, x1.x, x2.w);
                    floatfloat_fms(val_hi, val_lo, x1.y, x2.z);
                    floatfloat_fma(val_hi, val_lo, x1.z, x2.y);
                    floatfloat_fma(val_hi, val_lo, x1.w, x2.x);
                    floatfloat_hilo_add(hi.w, lo.w, val_hi, val_lo);
                }}";

                /// <summary>原子性保証加算</summary>
                public static string AtomicAdd =>
                $@"
                static __inline__ __device__ void floatfloat_atomicadd(float4 *ptr, float4 hi, float4 lo){{
                    float *ptr_float = (float*)ptr;

                    float tmpx = atomicAdd(ptr_float, hi.x);
                    atomicAdd(ptr_float + 1, lo.x - (((tmpx + hi.x) - tmpx) - hi.x));
                    float tmpy = atomicAdd(ptr_float + 2, hi.y);
                    atomicAdd(ptr_float + 3, lo.y - (((tmpy + hi.y) - tmpy) - hi.y));
                    float tmpz = atomicAdd(ptr_float + 4, hi.z);
                    atomicAdd(ptr_float + 5, lo.z - (((tmpz + hi.z) - tmpz) - hi.z));
                    float tmpw = atomicAdd(ptr_float + 6, hi.w);
                    atomicAdd(ptr_float + 7, lo.w - (((tmpw + hi.w) - tmpw) - hi.w));
                }}";
            }

            /// <summary>3次元ベクトル</summary>
            public static class Trivector {
                /// <summary>カーネル積</summary>
                public static string KernelProd =>
                $@"
                static __inline__ __device__ void trivector_quaternion_kernelprod(float4 &hi, float4 &lo, float3 v, float3 u, float4 q){{
                    float vxqx = v.x * q.x, vxqy = v.x * q.y, vxqz = v.x * q.z, vxqw = v.x * q.w;
                    float vyqx = v.y * q.x, vyqy = v.y * q.y, vyqz = v.y * q.z, vyqw = v.y * q.w;
                    float vzqx = v.z * q.x, vzqy = v.z * q.y, vzqz = v.z * q.z, vzqw = v.z * q.w;

                    float val_hi, val_lo;

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, u.x, (vzqz + vxqx - vyqw));
                    floatfloat_fma(val_hi, val_lo, u.y, (vxqw + vyqx - vzqy));
                    floatfloat_fma(val_hi, val_lo, u.z, (vyqy + vzqx - vxqz));
                    floatfloat_hilo_add(hi.x, lo.x, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, u.x, (vzqw + vxqy + vyqz));
                    floatfloat_fma(val_hi, val_lo, u.y, (vxqz - vyqy - vzqx));
                    floatfloat_fma(val_hi, val_lo, u.z, (vyqx - vzqy + vxqw));
                    floatfloat_hilo_add(hi.y, lo.y, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, u.x, (vzqx - vxqz + vyqy));
                    floatfloat_fma(val_hi, val_lo, u.y, (vxqy + vyqz + vzqw));
                    floatfloat_fma(val_hi, val_lo, u.z, (vyqw - vzqz - vxqx));
                    floatfloat_hilo_add(hi.z, lo.z, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, u.x, (vzqy - vxqw - vyqx));
                    floatfloat_fma(val_hi, val_lo, u.y, (vxqx - vyqw + vzqz));
                    floatfloat_fma(val_hi, val_lo, u.z, (vyqz + vzqw + vxqy));
                    floatfloat_hilo_add(hi.w, lo.w, val_hi, val_lo);
                }}";

                /// <summary>積</summary>
                public static string Mul =>
                $@"
                static __inline__ __device__ void trivector_quaternion_mul(float3 &hi, float3 &lo, float3 v, float4 q){{
                    float sx = q.x * q.x, sy = q.y * q.y, sz = q.z * q.z, sw = q.w * q.w;
                    float mx = q.y * q.z, my = q.z * q.w, mz = q.w * q.y;
                    float nx = q.x * q.y, ny = q.x * q.z, nz = q.x * q.w;
                    float vx2 = ldexpf(v.x, 1), vy2 = ldexpf(v.y, 1), vz2 = ldexpf(v.z, 1);

                    float val_hi, val_lo;

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, v.x, (sx + sy - sz - sw));
                    floatfloat_fma(val_hi, val_lo, vy2, (mx - nz));
                    floatfloat_fma(val_hi, val_lo, vz2, (mz + ny));
                    floatfloat_hilo_add(hi.x, lo.x, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, v.y, (sx - sy + sz - sw));
                    floatfloat_fma(val_hi, val_lo, vz2, (my - nx));
                    floatfloat_fma(val_hi, val_lo, vx2, (mx + nz));
                    floatfloat_hilo_add(hi.y, lo.y, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, v.z, (sx - sy - sz + sw));
                    floatfloat_fma(val_hi, val_lo, vx2, (mz - ny));
                    floatfloat_fma(val_hi, val_lo, vy2, (my + nx));
                    floatfloat_hilo_add(hi.z, lo.z, val_hi, val_lo);
                }}";

                /// <summary>積勾配</summary>
                public static string MulGrad =>
                $@"
                static __inline__ __device__ void trivector_quaternion_mulgrad(float3 &hi, float3 &lo, float3 v, float4 q){{
                    float sx = q.x * q.x, sy = q.y * q.y, sz = q.z * q.z, sw = q.w * q.w;
                    float mx = q.y * q.z, my = q.z * q.w, mz = q.w * q.y;
                    float nx = q.x * q.y, ny = q.x * q.z, nz = q.x * q.w;
                    float vx2 = ldexpf(v.x, 1), vy2 = ldexpf(v.y, 1), vz2 = ldexpf(v.z, 1);

                    float val_hi, val_lo;

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, v.x, (sx + sy - sz - sw));
                    floatfloat_fma(val_hi, val_lo, vy2, (mx + nz));
                    floatfloat_fma(val_hi, val_lo, vz2, (mz - ny));
                    floatfloat_hilo_add(hi.x, lo.x, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, v.y, (sx - sy + sz - sw));
                    floatfloat_fma(val_hi, val_lo, vz2, (my + nx));
                    floatfloat_fma(val_hi, val_lo, vx2, (mx - nz));
                    floatfloat_hilo_add(hi.y, lo.y, val_hi, val_lo);

                    val_hi = 0.0; val_lo = 0.0;
                    floatfloat_fma(val_hi, val_lo, v.z, (sx - sy - sz + sw));
                    floatfloat_fma(val_hi, val_lo, vx2, (mz + ny));
                    floatfloat_fma(val_hi, val_lo, vy2, (my - nx));
                    floatfloat_hilo_add(hi.z, lo.z, val_hi, val_lo);
                }}";
            }
        }
    }
}
