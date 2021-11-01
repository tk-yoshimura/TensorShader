using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>水平加算</summary>
    public sealed class HorizontalAdd : Shader {
        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public HorizontalAdd() {
            string code = $@"

            __global__ void horizontal_add(const float* __restrict__ inmap, float* __restrict__ outmap, unsigned int n) {{

                unsigned int i = {Defines.IndexX};

                if(i >= n){{
                    return;
                }}

                outmap[i] = inmap[i * 2] + inmap[i * 2 + 1];
            }}";

            this.Kernel = new Kernel(code, "horizontal_add");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            uint n = (args.Last() as uint?).Value;

            Kernel.Execute(n, dynamic_shared_memory_bytes: 0, stream, args);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 3) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[2] is not uint n || n < 1) {
                throw new ArgumentException(nameof(n));
            }

            if (args[0] is not CudaArray<float> inmap || inmap.Length < n * 2) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < n) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
