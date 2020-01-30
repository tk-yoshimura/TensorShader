using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Trivector.Arithmetric {

    /// <summary>回転積四元数勾配</summary>
    public sealed class QuaternionMulQGrad : Shader {
        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public QuaternionMulQGrad() {
            string code = $@"

            __global__ void trivector_quaternion_mulqgrad(const float3* __restrict__ inmap1_vector,
                                                          const float3* __restrict__ inmap2_vector,
                                                          const float4* __restrict__ inmap_quaternion,
                                                          float4* __restrict__ outmap_quaternion,
                                                          unsigned int n) {{

                unsigned int i = {Defines.IndexX};
                if (i >= n) {{
                    return;
                }}

                float3 v = inmap1_vector[i], u = inmap2_vector[i];
                float4 p, q = inmap_quaternion[i];

                float vxqx = v.x * q.x, vxqy = v.x * q.y, vxqz = v.x * q.z, vxqw = v.x * q.w;
                float vyqx = v.y * q.x, vyqy = v.y * q.y, vyqz = v.y * q.z, vyqw = v.y * q.w;
                float vzqx = v.z * q.x, vzqy = v.z * q.y, vzqz = v.z * q.z, vzqw = v.z * q.w;

                p.x = ldexpf(u.x * (vzqz + vxqx - vyqw) + u.y * (vxqw + vyqx - vzqy) + u.z * (vyqy + vzqx - vxqz), 1);
                p.y = ldexpf(u.x * (vzqw + vxqy + vyqz) + u.y * (vxqz - vyqy - vzqx) + u.z * (vyqx - vzqy + vxqw), 1);
                p.z = ldexpf(u.x * (vzqx - vxqz + vyqy) + u.y * (vxqy + vyqz + vzqw) + u.z * (vyqw - vzqz - vxqx), 1);
                p.w = ldexpf(u.x * (vzqy - vxqw - vyqx) + u.y * (vxqx - vyqw + vzqz) + u.z * (vyqz + vzqw + vxqy), 1);

                outmap_quaternion[i] = p;
            }}";

            this.Kernel = new Kernel(code, "trivector_quaternion_mulqgrad");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> v = args[0] as CudaArray<float>;
            CudaArray<float> u = args[1] as CudaArray<float>;
            CudaArray<float> q = args[2] as CudaArray<float>;
            CudaArray<float> p = args[3] as CudaArray<float>;

            uint length = (args.Last() as uint?).Value;
            uint n = length / 3;

            Kernel.Execute(n, dynamic_shared_memory_bytes: 0, stream, v, u, q, p, n);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[4] is uint vector_length) || vector_length % 3 != 0) {
                throw new ArgumentException(nameof(vector_length));
            }

            uint quaternion_length = vector_length / 3 * 4;

            if (!(args[0] is CudaArray<float> varr) || varr.Length < vector_length) {
                throw new ArgumentException(nameof(varr));
            }

            if (!(args[1] is CudaArray<float> uarr) || uarr.Length < vector_length) {
                throw new ArgumentException(nameof(uarr));
            }

            if (!(args[2] is CudaArray<float> qarr) || qarr.Length < quaternion_length) {
                throw new ArgumentException(nameof(qarr));
            }

            if (!(args[3] is CudaArray<float> parr) || parr.Length < quaternion_length) {
                throw new ArgumentException(nameof(parr));
            }
        }
    }
}
