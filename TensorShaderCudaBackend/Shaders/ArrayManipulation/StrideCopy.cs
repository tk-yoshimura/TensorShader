using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>ストライドコピー</summary>
    public sealed class StrideCopy : Shader {

        /// <summary>ストライド</summary>
        public uint Stride { private set; get; }

        /// <summary>インデクス</summary>
        public uint Index { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Stride)}={Stride} {nameof(Index)}={Index}";

        /// <summary>コンストラクタ</summary>
        public StrideCopy(uint stride, uint index) {
            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            if (index >= stride) {
                throw new ArgumentException(nameof(index));
            }

            this.Stride = stride;
            this.Index = index;

            string code = $@"

            __global__ void stride_copy(float *x, float *y, unsigned int n) {{
                unsigned int i = {Defines.IndexX};
                if (i >= n) {{
                    return;
                }}

                y[i * {Stride}] = x[i];
            }}";

            this.Kernel = new Kernel(code, "stride_copy");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> x = args[0] as CudaArray<float>;
            CudaArray<float> y = args[1] as CudaArray<float>;
            uint length = (args.Last() as uint?).Value;
            uint n = length / Stride;

            y.ZerosetAsync(stream, length);

            Kernel.Execute(n, dynamic_shared_memory_bytes: 0, stream, x, y.ElementPtr(Index), n);
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 3) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint length) || length % Stride != 0) {
                throw new ArgumentException(nameof(length));
            }

            uint n = length / Stride;

            if (!(args[0] is CudaArray<float> xarr) || xarr.Length < n) {
                throw new ArgumentException(nameof(xarr));
            }

            if (!(args[1] is CudaArray<float> yarr) || yarr.Length < length) {
                throw new ArgumentException(nameof(yarr));
            }
        }
    }
}
