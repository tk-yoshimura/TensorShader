using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Channelwise {

    /// <summary>チャネル独立演算</summary>
    public abstract class Channelwise : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>マップ配列数</summary>
        public int MapArrays { private set; get; }

        /// <summary>関数名</summary>
        public string FuncName { private set; get; }

        /// <summary>定数メモリを使用するか</summary>
        public bool UseConstMemory { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {FuncName} {nameof(Channels)} = {Channels}";

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="map_arrays">マップ配列数</param>
        /// <param name="name">関数名</param>
        public Channelwise(uint channels, int map_arrays, string name) {
            if (channels < 1) {
                throw new ArgumentException(nameof(channels));
            }
            if (map_arrays < 1) {
                throw new ArgumentException(nameof(map_arrays));
            }

            this.Channels = channels;
            this.MapArrays = map_arrays;
            this.FuncName = $"{name} {nameof(channels)}={channels}";
            this.UseConstMemory = channels <= 4;
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            uint length = (args.Last() as uint?).Value;

            if (UseConstMemory) {
                CudaArray<float> v = args[0] as CudaArray<float>;

                if (stream != null) {
                    Kernel.StoreConstMemoryAsync(stream, "v", v, Channels);
                }
                else {
                    Kernel.StoreConstMemory("v", v, Channels);
                }

                args = args.Skip(1).ToArray();
            }

            Kernel.Execute(length, dynamic_shared_memory_bytes: 0, stream, args);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != MapArrays + 2) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[MapArrays + 1] is uint length) || length < 0) {
                throw new ArgumentException(nameof(length));
            }


            if (!(args[0] is CudaArray<float> vectorarr) || vectorarr.Length < (ulong)Channels) {
                throw new ArgumentException(nameof(vectorarr));
            }

            for (int i = 1; i < MapArrays + 1; i++) {
                if (!(args[i] is CudaArray<float> arr) || arr.Length < length) {
                    throw new ArgumentException($"{nameof(arr)}[{i}]");
                }
            }
        }
    }
}
