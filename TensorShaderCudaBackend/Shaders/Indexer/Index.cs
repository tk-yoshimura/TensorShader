using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Indexer {

    /// <summary>Index</summary>
    public sealed class Index : Shader {

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public Index() {
            string code = $@"

            __global__ void index(float* __restrict__ y, unsigned int round, unsigned int stride, unsigned axislength) {{
                unsigned int i = {Defines.IndexX};
                if (i >= round) {{
                    return;
                }}

                y[i] = (float)((i / stride) % axislength);
            }}";

            this.Kernel = new Kernel(code, "index");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> y = args[0] as CudaArray<float>;
            uint length = (args[1] as uint?).Value;
            uint stride = (args[2] as uint?).Value;
            uint axislength = (args[3] as uint?).Value;

            uint round = stride * axislength;
            while (round <= 1024 && round * 2 <= length) {
                round *= 2;
            }

            Kernel.Execute(
                round,
                dynamic_shared_memory_bytes: 0,
                stream,
                y,
                round, stride, axislength
            );

            while (round * 2 <= length) {
                y.CopyToAsync(stream, 0, y, round, round);
                round *= 2;
            }
            if (round < length) {
                y.CopyToAsync(stream, 0, y, round, length - round);
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint stride) || stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            if (!(args[3] is uint axislength) || axislength < 1) {
                throw new ArgumentException(nameof(axislength));
            }

            if (!(args[1] is uint length) || length % (stride * axislength) != 0) {
                throw new ArgumentException(nameof(length));
            }

            if (!(args[0] is CudaArray<float> y) || y.Length < length) {
                throw new ArgumentException(nameof(y));
            }
        }
    }
}
