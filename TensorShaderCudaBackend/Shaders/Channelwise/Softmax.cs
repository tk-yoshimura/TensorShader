using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Channelwise {

    /// <summary>Softmax</summary>
    public sealed class Softmax : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels}";

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public Softmax(uint channels) {
            if (channels < 1) {
                throw new ArgumentException(null, nameof(channels));
            }

            this.Channels = channels;

            string code = $@"
            __global__ void softmax(const float* __restrict__ x, float* __restrict__ y, unsigned int indexes) {{
                unsigned int idx = {Defines.IndexX};
                if (idx >= indexes) {{
                    return;
                }}

                float vsum = 0;

                for(int i = 0, map_idx = idx * {Channels}; i < {Channels}; i++, map_idx++){{
                    float v = expf(x[map_idx]);
                    y[map_idx] = v;
                    vsum += v;
                }}

                float vscale = 1 / vsum;

                for(int i = 0, map_idx = idx * {Channels}; i < {Channels}; i++, map_idx++){{
                    y[map_idx] *= vscale;
                }}
            }}";

            this.Kernel = new Kernel(code, "softmax");
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
