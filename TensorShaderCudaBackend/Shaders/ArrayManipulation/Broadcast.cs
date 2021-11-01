using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>ブロードキャスト</summary>
    public sealed class Broadcast : Shader {

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>実行あたりのスライド数</summary>
        public static uint SlidesPerExecute => 0x8000;

        /// <summary>コンストラクタ</summary>
        public Broadcast() {
            string code = $@"

            __global__ void broadcast(const float* __restrict__ inmap, float* __restrict__ outmap,
                                      unsigned int inmap_stride, unsigned int outmap_stride,
                                      unsigned int slides) {{

                unsigned int i = {Defines.IndexX}, j = {Defines.IndexY};
                if (i >= outmap_stride || j >= slides) {{
                    return;
                }}

                unsigned int inmap_idx = i % inmap_stride + j * inmap_stride;
                unsigned int outmap_idx = i + j * outmap_stride;

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "broadcast");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;

            uint inmap_stride = (args[2] as uint?).Value;
            uint outmap_stride = (args[3] as uint?).Value;
            uint slides = (args[4] as uint?).Value;

            for (uint s = 0; s < slides; s += SlidesPerExecute) {
                uint sl = Math.Min(SlidesPerExecute, slides - s);

                Kernel.Execute(
                    indexes: (outmap_stride, sl),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(s * inmap_stride),
                    outmap.ElementPtr(s * outmap_stride),
                    inmap_stride, outmap_stride, sl
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 5) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[2] is not uint inmap_stride || inmap_stride < 1) {
                throw new ArgumentException(nameof(inmap_stride));
            }

            if (args[3] is not uint outmap_stride || outmap_stride < 1) {
                throw new ArgumentException(nameof(outmap_stride));
            }

            if (args[4] is not uint slides) {
                throw new ArgumentException(nameof(slides));
            }

            uint inmap_length = inmap_stride * slides;
            uint outmap_length = outmap_stride * slides;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < inmap_length) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < outmap_length) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
