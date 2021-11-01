using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>指数付き要素独立演算</summary>
    public abstract class FactorElementwise : Shader {

        /// <summary>指数数</summary>
        public int Factors { private set; get; }

        /// <summary>配列数</summary>
        public int Arrays { private set; get; }

        /// <summary>関数名</summary>
        public string FuncName { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {FuncName}";

        /// <summary>コンストラクタ</summary>
        /// <param name="factors">指数数</param>
        /// <param name="arrays">配列数</param>
        /// <param name="name">関数名</param>
        public FactorElementwise(int factors, int arrays, string name) {
            if (factors < 1) {
                throw new ArgumentException(null, nameof(factors));
            }
            if (arrays < 1) {
                throw new ArgumentException(null, nameof(arrays));
            }

            this.Factors = factors;
            this.Arrays = arrays;
            this.FuncName = name;
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            uint length = (args.Last() as uint?).Value;

            if (Factors == 1) {
                CudaArray<float> c = args[0] as CudaArray<float>;

                if (stream is not null) {
                    Kernel.StoreConstMemoryAsync(stream, "c", c, 1);
                }
                else {
                    Kernel.StoreConstMemory("c", c, 1);
                }
            }
            else {
                for (int i = 0; i < Factors; i++) {
                    CudaArray<float> c = args[i] as CudaArray<float>;

                    if (stream is not null) {
                        Kernel.StoreConstMemoryAsync(stream, $"c{i + 1}", c, 1);
                    }
                    else {
                        Kernel.StoreConstMemory($"c{i + 1}", c, 1);
                    }
                }
            }

            args = args.Skip(Factors).ToArray();

            Kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream, args);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != Factors + Arrays + 1) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[Factors + Arrays] is not uint length) {
                throw new ArgumentException(nameof(length));
            }

            for (int i = 0; i < Factors; i++) {
                if (args[i] is not CudaArray<float> arr || arr.Length < 1) {
                    throw new ArgumentException($"{nameof(arr)}[{i}]");
                }
            }

            for (int i = Factors; i < Factors + Arrays; i++) {
                if (args[i] is not CudaArray<float> arr || arr.Length < length) {
                    throw new ArgumentException($"{nameof(arr)}[{i}]");
                }
            }
        }
    }
}
