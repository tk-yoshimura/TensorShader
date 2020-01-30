using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>集約</summary>
    public sealed class Gather : Shader {

        /// <summary>集約アレイ数</summary>
        public uint Arrays { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Arrays)}={Arrays}";

        /// <summary>コンストラクタ</summary>
        public Gather(uint arrays) {
            if (arrays < 2) {
                throw new ArgumentException(nameof(arrays));
            }

            this.Arrays = arrays;

            string code = $@"

            __global__ void gather(const float* __restrict__ inmap, float* __restrict__ outmap, unsigned int n) {{
                unsigned int i = {Defines.IndexX};
                if (i >= n) {{
                    return;
                }}

                outmap[i * {Arrays}] = inmap[i];
            }}";

            this.Kernel = new Kernel(code, "gather");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> y = args[Arrays] as CudaArray<float>;
            uint length = (args.Last() as uint?).Value;
            uint n = length / Arrays;

            for (uint i = 0; i < Arrays; i++) {
                CudaArray<float> x = args[i] as CudaArray<float>;

                Kernel.Execute(n, dynamic_shared_memory_bytes: 0, stream, x, y.ElementPtr(i), n);
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != Arrays + 2) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[Arrays + 1] is uint length) || length % Arrays != 0) {
                throw new ArgumentException(nameof(length));
            }

            uint n = length / Arrays;

            for (uint i = 0; i < Arrays; i++) {
                if (!(args[i] is CudaArray<float> xarr) || xarr.Length < n) {
                    throw new ArgumentException($"{nameof(xarr)}[{i}]");
                }
            }

            if (!(args[Arrays] is CudaArray<float> yarr) || yarr.Length < length) {
                throw new ArgumentException(nameof(yarr));
            }
        }
    }
}
