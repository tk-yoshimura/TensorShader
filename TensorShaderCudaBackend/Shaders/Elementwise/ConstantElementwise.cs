using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>定数付き要素独立演算</summary>
    public abstract class ConstantElementwise : Shader {

        /// <summary>定数数</summary>
        public int Constants { private set; get; }

        /// <summary>配列数</summary>
        public int Arrays { private set; get; }

        /// <summary>関数名</summary>
        public string FuncName { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {FuncName}";

        /// <summary>コンストラクタ</summary>
        /// <param name="constants">定数数</param>
        /// <param name="arrays">配列数</param>
        /// <param name="name">関数名</param>
        public ConstantElementwise(int constants, int arrays, string name) {
            if (constants < 1) {
                throw new ArgumentException(nameof(constants));
            }
            if (arrays < 1) {
                throw new ArgumentException(nameof(arrays));
            }

            this.Constants = constants;
            this.Arrays = arrays;
            this.FuncName = name;
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            uint length = (args.Last() as uint?).Value;

            Kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream, args);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != Constants + Arrays + 1) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[Constants + Arrays] is uint length)) {
                throw new ArgumentException(nameof(length));
            }

            for (int i = 0; i < Constants; i++) {
                if (!(args[i] is float)) {
                    throw new ArgumentException("const val");
                }
            }

            for (int i = Constants; i < Constants + Arrays; i++) {
                if (!(args[i] is CudaArray<float> arr) || arr.Length < length) {
                    throw new ArgumentException($"{nameof(arr)}[{i}]");
                }
            }
        }
    }
}
