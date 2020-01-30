using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>初期化</summary>
    public sealed class Clear : Shader {

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public Clear() {
            string code = $@"

            __global__ void clear(float* __restrict__ refmap, float c, unsigned int length) {{
                unsigned int i = {Defines.IndexX};
                if (i >= length) {{
                    return;
                }}

                refmap[i] = c;
            }}";

            this.Kernel = new Kernel(code, "clear");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            uint length = (args.Last() as uint?).Value;

            Kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream, args);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 3) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint length)) {
                throw new ArgumentException(nameof(length));
            }

            if (!(args[0] is CudaArray<float> arr) || arr.Length < length) {
                throw new ArgumentException(nameof(arr));
            }

            if (!(args[1] is float)) {
                throw new ArgumentException("const val");
            }
        }
    }
}
