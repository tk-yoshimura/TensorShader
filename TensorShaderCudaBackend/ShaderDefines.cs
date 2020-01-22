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
            public static string StoreSharedMemory(string elem, uint length, uint threads){
                if(threads > length){
                        return $@"
                        static __inline__ __device__ void store_smem({elem} *ptr, {elem} *smem, unsigned int thread_idx){{
                            if(thread_idx < {length}) smem[thread_idx] = ptr[thread_idx];
                            __syncthreads();
                        }}";
                } 
                else if(threads == length){ 
                        return $@"
                        static __inline__ __device__ void store_smem({elem} *ptr, {elem} *smem, unsigned int thread_idx){{
                            smem[thread_idx] = ptr[thread_idx];
                            __syncthreads();
                        }}";
                }
                else if(threads * 8 >= length){
                    if(length % threads == 0) { 
                        return $@"
                        static __inline__ __device__ void store_smem({elem} *ptr, {elem} *smem, unsigned int thread_idx){{
                            unsigned int i = thread_idx;
                            { string.Join(" ", Enumerable.Repeat($"smem[i] = ptr[i]; i += {threads};", (int)(length / threads))) }
                            __syncthreads();
                        }}";
                    }
                    else { 
                        return $@"
                        static __inline__ __device__ void store_smem({elem} *ptr, {elem} *smem, unsigned int thread_idx){{
                            unsigned int i = thread_idx;
                            { string.Join(" ", Enumerable.Repeat($"smem[i] = ptr[i]; i += {threads};", (int)(length / threads))) }
                            if(i < {length}) smem[i] = ptr[i];
                            __syncthreads();
                        }}";
                    }
                }
                else{
                        return $@"
                        static __inline__ __device__ void store_smem({elem} *ptr, {elem} *smem, unsigned int thread_idx){{ 
                            for(unsigned int i = thread_idx; i < {length}; i += {threads}){{
                                smem[i] = ptr[i];
                            }}
                            __syncthreads();
                        }}";
                }
            }

            /// <summary>複素数</summary>
            public static class Complex {
                /// <summary>カーネル積</summary>
                public static string KernelProd =>
                $@"
                static __inline__ __device__ void complex_kernelprod(float2 &hi, float2 &lo, float2 x1, float2 x2){{
                    floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                    floatfloat_add(hi.x, lo.x, x1.y * x2.y);
                    floatfloat_sub(hi.y, lo.y, x1.y * x2.x);
                    floatfloat_add(hi.y, lo.y, x1.x * x2.y);
                }}";

                /// <summary>積</summary>
                public static string Mul =>
                $@"
                static __inline__ __device__ void complex_mul(float2 &hi, float2 &lo, float2 x1, float2 x2){{
                    floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                    floatfloat_sub(hi.x, lo.x, x1.y * x2.y);
                    floatfloat_add(hi.y, lo.y, x1.x * x2.y);
                    floatfloat_add(hi.y, lo.y, x1.y * x2.x);
                }}";

                /// <summary>積勾配</summary>
                public static string MulGrad =>
                $@"
                static __inline__ __device__ void complex_mulgrad(float2 &hi, float2 &lo, float2 x1, float2 x2){{
                    floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                    floatfloat_add(hi.x, lo.x, x1.y * x2.y);
                    floatfloat_add(hi.y, lo.y, x1.y * x2.x);
                    floatfloat_sub(hi.y, lo.y, x1.x * x2.y);
                }}";

                /// <summary>原子性保証加算</summary>
                public static string AtomicAdd =>
                $@"
                static __inline__ __device__ void floatfloat_atomicadd(float2 *ptr, float2 hi, float2 lo){{
                    float *ptr_float = (float*)(void*)ptr;

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
                    floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                    floatfloat_add(hi.x, lo.x, x1.y * x2.y);
                    floatfloat_add(hi.x, lo.x, x1.z * x2.z);
                    floatfloat_add(hi.x, lo.x, x1.w * x2.w);

                    floatfloat_add(hi.y, lo.y, x1.x * x2.y);
                    floatfloat_sub(hi.y, lo.y, x1.y * x2.x);
                    floatfloat_sub(hi.y, lo.y, x1.z * x2.w);
                    floatfloat_add(hi.y, lo.y, x1.w * x2.z);

                    floatfloat_add(hi.z, lo.z, x1.x * x2.z);
                    floatfloat_add(hi.z, lo.z, x1.y * x2.w);
                    floatfloat_sub(hi.z, lo.z, x1.z * x2.x);
                    floatfloat_sub(hi.z, lo.z, x1.w * x2.y);

                    floatfloat_add(hi.w, lo.w, x1.x * x2.w);
                    floatfloat_sub(hi.w, lo.w, x1.y * x2.z);
                    floatfloat_add(hi.w, lo.w, x1.z * x2.y);
                    floatfloat_sub(hi.w, lo.w, x1.w * x2.x);
                }}";

                /// <summary>積</summary>
                public static string Mul =>
                $@"
                static __inline__ __device__ void quaternion_mul(float4 &hi, float4 &lo, float4 x1, float4 x2){{
                    floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                    floatfloat_sub(hi.x, lo.x, x1.y * x2.y);
                    floatfloat_sub(hi.x, lo.x, x1.z * x2.z);
                    floatfloat_sub(hi.x, lo.x, x1.w * x2.w);

                    floatfloat_add(hi.y, lo.y, x1.x * x2.y);
                    floatfloat_add(hi.y, lo.y, x1.y * x2.x);
                    floatfloat_add(hi.y, lo.y, x1.z * x2.w);
                    floatfloat_sub(hi.y, lo.y, x1.w * x2.z);

                    floatfloat_add(hi.z, lo.z, x1.x * x2.z);
                    floatfloat_sub(hi.z, lo.z, x1.y * x2.w);
                    floatfloat_add(hi.z, lo.z, x1.z * x2.x);
                    floatfloat_add(hi.z, lo.z, x1.w * x2.y);

                    floatfloat_add(hi.w, lo.w, x1.x * x2.w);
                    floatfloat_add(hi.w, lo.w, x1.y * x2.z);
                    floatfloat_sub(hi.w, lo.w, x1.z * x2.y);
                    floatfloat_add(hi.w, lo.w, x1.w * x2.x);
                }}";

                /// <summary>積勾配</summary>
                public static string MulGrad =>
                $@"
                static __inline__ __device__ void quaternion_mulgrad(float4 &hi, float4 &lo, float4 x1, float4 x2){{
                    floatfloat_add(hi.x, lo.x, x1.x * x2.x);
                    floatfloat_add(hi.x, lo.x, x1.y * x2.y);
                    floatfloat_add(hi.x, lo.x, x1.z * x2.z);
                    floatfloat_add(hi.x, lo.x, x1.w * x2.w);

                    floatfloat_sub(hi.y, lo.y, x1.x * x2.y);
                    floatfloat_add(hi.y, lo.y, x1.y * x2.x);
                    floatfloat_sub(hi.y, lo.y, x1.z * x2.w);
                    floatfloat_add(hi.y, lo.y, x1.w * x2.z);

                    floatfloat_sub(hi.z, lo.z, x1.x * x2.z);
                    floatfloat_add(hi.z, lo.z, x1.y * x2.w);
                    floatfloat_add(hi.z, lo.z, x1.z * x2.x);
                    floatfloat_sub(hi.z, lo.z, x1.w * x2.y);

                    floatfloat_sub(hi.w, lo.w, x1.x * x2.w);
                    floatfloat_sub(hi.w, lo.w, x1.y * x2.z);
                    floatfloat_add(hi.w, lo.w, x1.z * x2.y);
                    floatfloat_add(hi.w, lo.w, x1.w * x2.x);
                }}";

                /// <summary>原子性保証加算</summary>
                public static string AtomicAdd =>
                $@"
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
                }}";

                /// <summary>積</summary>
                public static string Mul =>
                $@"
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
                }}";

                /// <summary>積勾配</summary>
                public static string MulGrad =>
                $@"
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
                }}";
            }
        }
    }
}
