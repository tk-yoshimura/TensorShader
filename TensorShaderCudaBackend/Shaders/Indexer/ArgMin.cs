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
            __global__ void argmin(float *x, float *v, unsigned int indexes) {{
                unsigned int i = {Defines.IndexX};
                if (i >= indexes) {{
                    return;
                }}

                unsigned int inmap_idx = i * {Channels};

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

                v[i] = (float)vmin_i;
            }}";

            this.Kernel = new Kernel(code, "argmin");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            uint indexes = (args.Last() as uint?).Value;

            Kernel.Execute(
                indexes, dynamic_shared_memory_bytes: 0, stream, 
                new object[]{ args[0], args[1], args[3] });
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 4) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint length)) {
                throw new ArgumentException($"{nameof(args)}[2]");
            }

            if (!(args[3] is uint indexes) || length != indexes * Channels) {
                throw new ArgumentException($"{nameof(args)}[3]");
            }

            if (!(args[0] is CudaArray<float> x) || x.Length < (ulong)length) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if (!(args[1] is CudaArray<float> v) || v.Length < (ulong)indexes) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }
        }
    }
}
