using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Channelwise {

    /// <summary>Hardmax</summary>
    public sealed class Hardmax : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels}";

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public Hardmax(uint channels) {
            if (channels < 1) {
                throw new ArgumentException(null, nameof(channels));
            }

            this.Channels = channels;

            string code = $@"
            __global__ void hardmax(const float* __restrict__ x, float* __restrict__ y, unsigned int indexes) {{
                unsigned int idx = {Defines.IndexX};
                if (idx >= indexes) {{
                    return;
                }}

                x += idx * {Channels};
                y += idx * {Channels};

                float vmax = x[0];
                int vmax_i = 0;

                for(int i = 1; i < {Channels}; i++){{
                    float v = x[i];
                    if(v > vmax){{
                        vmax = v;
                        vmax_i = i;
                    }}
                }}

                y[vmax_i] = 1;
            }}";

            this.Kernel = new Kernel(code, "hardmax");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> x = args[0] as CudaArray<float>;
            CudaArray<float> y = args[1] as CudaArray<float>;
            uint length = (args[2] as uint?).Value;
            uint indexes = (args.Last() as uint?).Value;

            y.Zeroset(length);

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
                throw new ArgumentException(null, nameof(args));
            }

            if (args[2] is not uint length) {
                throw new ArgumentException(nameof(length));
            }

            if (args[3] is not uint indexes || length != indexes * Channels) {
                throw new ArgumentException(nameof(indexes));
            }

            if (args[0] is not CudaArray<float> x || x.Length < length) {
                throw new ArgumentException(nameof(x));
            }

            if (args[1] is not CudaArray<float> y || y.Length < length) {
                throw new ArgumentException(nameof(y));
            }
        }
    }
}
