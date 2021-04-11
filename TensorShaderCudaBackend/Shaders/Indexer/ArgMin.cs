using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Indexer {

    /// <summary>ArgMin</summary>
    public sealed class ArgMin : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels}";

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public ArgMin(uint channels) {
            if (channels < 1) {
                throw new ArgumentException(nameof(channels));
            }

            this.Channels = channels;

            string code = $@"
            __global__ void argmin(const float* __restrict__ x, float* __restrict__ y, unsigned int indexes) {{
                unsigned int idx = {Defines.IndexX};
                if (idx >= indexes) {{
                    return;
                }}

                unsigned int inmap_idx = idx * {Channels};

                float vmin = x[inmap_idx];
                int vmin_i = 0;

                for(int i = 1; i < {Channels}; i++){{
                    inmap_idx++;

                    float v = x[inmap_idx];
                    if(v < vmin){{
                        vmin = v;
                        vmin_i = i;
                    }}
                }}

                y[idx] = (float)vmin_i;
            }}";

            this.Kernel = new Kernel(code, "argmin");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> x = args[0] as CudaArray<float>;
            CudaArray<float> y = args[1] as CudaArray<float>;
            uint indexes = (args.Last() as uint?).Value;

            Kernel.Execute(
                indexes,
                dynamic_shared_memory_bytes: 0,
                stream,
                x, y, indexes
            );
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint length)) {
                throw new ArgumentException(nameof(length));
            }

            if (!(args[3] is uint indexes) || length != indexes * Channels) {
                throw new ArgumentException(nameof(indexes));
            }

            if (!(args[0] is CudaArray<float> x) || x.Length < length) {
                throw new ArgumentException(nameof(x));
            }

            if (!(args[1] is CudaArray<float> y) || y.Length < indexes) {
                throw new ArgumentException(nameof(y));
            }
        }
    }
}
