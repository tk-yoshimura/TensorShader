using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Trivector.Arithmetric {

    /// <summary>回転積ベクトル勾配</summary>
    public sealed class QuaternionMulVGrad : Shader {
        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public QuaternionMulVGrad() {
            string code = $@"

            __global__ void trivector_quaternion_mulvgrad(float3 *inmap_vector,
                                                          float4 *inmap_quaternion,
                                                          float3 *outmap_vector,
                                                          unsigned int n) {{

                unsigned int i = {Defines.IndexX};
                if (i >= n) {{
                    return;
                }}

                float3 u, v = inmap_vector[i];
                float4 q = inmap_quaternion[i];

                float sx = q.x * q.x, sy = q.y * q.y, sz = q.z * q.z, sw = q.w * q.w;
                float mx = q.y * q.z, my = q.z * q.w, mz = q.w * q.y;
                float nx = q.x * q.y, ny = q.x * q.z, nz = q.x * q.w;

                u.x = v.x * (sx + sy - sz - sw) + ldexpf(v.y * (mx + nz) + v.z * (mz - ny), 1);
                u.y = v.y * (sx - sy + sz - sw) + ldexpf(v.z * (my + nx) + v.x * (mx - nz), 1);
                u.z = v.z * (sx - sy - sz + sw) + ldexpf(v.x * (mz + ny) + v.y * (my - nx), 1);

                outmap_vector[i] = u;
            }}";

            this.Kernel = new Kernel(code, "trivector_quaternion_mulvgrad");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> v = args[0] as CudaArray<float>;
            CudaArray<float> q = args[1] as CudaArray<float>;
            CudaArray<float> u = args[2] as CudaArray<float>;

            uint length = (args.Last() as uint?).Value;
            uint n = length / 3;

            Kernel.Execute(n, dynamic_shared_memory_bytes: 0, stream, v, q, u, n);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint vector_length) || vector_length % 3 != 0) {
                throw new ArgumentException(nameof(vector_length));
            }

            uint quaternion_length = vector_length / 3 * 4;

            if (!(args[0] is CudaArray<float> varr) || varr.Length < vector_length) {
                throw new ArgumentException(nameof(varr));
            }

            if (!(args[1] is CudaArray<float> qarr) || qarr.Length < quaternion_length) {
                throw new ArgumentException(nameof(qarr));
            }

            if (!(args[2] is CudaArray<float> uarr) || uarr.Length < vector_length) {
                throw new ArgumentException(nameof(uarr));
            }
        }
    }
}
