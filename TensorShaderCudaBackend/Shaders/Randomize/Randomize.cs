using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Randomize {

    /// <summary>乱数生成基底クラス</summary>
    public abstract class Randomize : Shader {

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>ワープサイズ</summary>
        public static uint WarpSize => 32;

        /// <summary>スレッドあたりの生成数</summary>
        public static uint RandomPerThread => 256;

        /// <summary>ワープあたりの生成数</summary>
        public static uint RandomPerWarp => 256 * WarpSize;

        /// <summary>乱数生成空回し</summary>
        public static int Dumps => 16;

        /// <summary>付与パラメータ数</summary>
        public int ExParams { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Randomize(int exparams = 0) {
            if (exparams < 0) {
                throw new ArgumentException(nameof(exparams));
            }

            this.ExParams = exparams;
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            Random random = args[2] as Random;

            byte[] bytes_random = new byte[sizeof(uint) * 3];
            random.NextBytes(bytes_random);

            uint seed1 = 0x1u | BitConverter.ToUInt32(bytes_random, 0);
            uint seed2 = 0x2u | BitConverter.ToUInt32(bytes_random, sizeof(uint));
            uint seed3 = 0x4u | BitConverter.ToUInt32(bytes_random, sizeof(uint) * 2);

            CudaArray<float> y = args[0] as CudaArray<float>;

            uint length = (args[1] as uint?).Value;

            uint warps = (length + RandomPerWarp - 1) / RandomPerWarp;

            object[] kernel_args = (new object[] { y, length, warps, seed1, seed2, seed3 }).Concat(args.Skip(3)).ToArray();

            Kernel.Execute(
                indexes: (WarpSize, warps),
                dynamic_shared_memory_bytes: 0,
                stream,
                kernel_args
            );
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 3 + ExParams) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[1] is uint length)) {
                throw new ArgumentException(nameof(length));
            }

            if (!(args[2] is Random random) || random == null) {
                throw new ArgumentException(nameof(random));
            }

            if (!(args[0] is CudaArray<float> arr) || arr.Length < length) {
                throw new ArgumentException(nameof(arr));
            }
        }
    }
}
