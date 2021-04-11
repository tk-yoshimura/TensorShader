using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Aggregation {

    /// <summary>集計基本クラス</summary>
    public abstract class Aggregation : Shader {

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>共有メモリ使用ライン数</summary>
        private uint SharedMemoryLines { set; get; }

        /// <summary>実行あたりのスライド数</summary>
        public static uint SlidesPerExecute => 0x8000;

        /// <summary>コンストラクタ</summary>
        protected Aggregation(uint shared_memory_lines) {
            if (shared_memory_lines < 1) {
                throw new ArgumentException(nameof(shared_memory_lines));
            }

            if (Kernel.MaxBlockSize > API.Cuda.CurrectDeviceProperty.SharedMemoryBytesPerBlock / shared_memory_lines / sizeof(float)) {
                throw new ArgumentException(nameof(shared_memory_lines));
            }

            this.SharedMemoryLines = shared_memory_lines;
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;

            uint inmap_stride = (args[2] as uint?).Value;
            uint outmap_stride = (args[3] as uint?).Value;
            uint slides = (args[4] as uint?).Value;

            uint axislength = inmap_stride / outmap_stride;

            uint max_shared_memory_length = Kernel.MaxBlockSize;
            uint shared_memory_length = 1;

            while (shared_memory_length * 2 < axislength) {

                shared_memory_length *= 2;

                if (shared_memory_length > max_shared_memory_length) {
                    shared_memory_length = max_shared_memory_length;
                    break;
                }
            }

            for (uint s = 0; s < slides; s += SlidesPerExecute) {
                uint sl = Math.Min(SlidesPerExecute, slides - s);

                Kernel.Execute(
                    indexes: (shared_memory_length * outmap_stride, sl),
                    block: (shared_memory_length, 1),
                    dynamic_shared_memory_bytes: shared_memory_length * SharedMemoryLines * sizeof(float),
                    stream,
                    inmap.ElementPtr(s * inmap_stride),
                    outmap.ElementPtr(s * outmap_stride),
                    axislength, shared_memory_length, outmap_stride, sl
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inmap_stride) || inmap_stride < 1) {
                throw new ArgumentException(nameof(inmap_stride));
            }

            if (!(args[3] is uint outmap_stride) || outmap_stride < 1 || inmap_stride % outmap_stride != 0) {
                throw new ArgumentException(nameof(outmap_stride));
            }

            if (!(args[4] is uint slides)) {
                throw new ArgumentException(nameof(slides));
            }

            uint inmap_length = inmap_stride * slides;
            uint outmap_length = outmap_stride * slides;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < inmap_length) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < outmap_length) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
