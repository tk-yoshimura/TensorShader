using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>抽出</summary>
    public sealed class Scatter : Shader {

        /// <summary>抽出アレイ数</summary>
        public uint Arrays { private set; get; }

        /// <summary>インデクス</summary>
        public uint Index { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Arrays)}={Arrays} {nameof(Index)}={Index}";

        /// <summary>コンストラクタ</summary>
        public Scatter(uint arrays, uint index) {
            if (arrays < 2) {
                throw new ArgumentException(nameof(arrays));
            }
            if (index >= arrays) {
                throw new ArgumentException(nameof(index));
            }

            this.Arrays = arrays;
            this.Index = index;

            string code = $@"

            __global__ void scatter(const float* __restrict__ inmap, float* __restrict__ outmap, unsigned int n) {{
                unsigned int i = {Defines.IndexX};
                if (i >= n) {{
                    return;
                }}

                outmap[i] = inmap[i * {Arrays}];
            }}";

            this.Kernel = new Kernel(code, "scatter");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> x = args[0] as CudaArray<float>;
            CudaArray<float> y = args[1] as CudaArray<float>;
            uint length = (args.Last() as uint?).Value;
            uint n = length / Arrays;

            Kernel.Execute(n, dynamic_shared_memory_bytes: 0, stream, x.ElementPtr(Index), y, n);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 3) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint length) || length % Arrays != 0) {
                throw new ArgumentException(nameof(length));
            }

            uint n = length / Arrays;

            if (!(args[0] is CudaArray<float> xarr) || xarr.Length < length) {
                throw new ArgumentException(nameof(xarr));
            }

            if (!(args[1] is CudaArray<float> yarr) || yarr.Length < n) {
                throw new ArgumentException(nameof(yarr));
            }
        }
    }
}
