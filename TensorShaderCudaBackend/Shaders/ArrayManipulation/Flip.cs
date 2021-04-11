using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>軸反転</summary>
    public sealed class Flip : Shader {

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>実行あたりのスライド数</summary>
        public static uint SlidesPerExecute => 0x8000;

        /// <summary>コンストラクタ</summary>
        public Flip() {
            string code = $@"

            __global__ void flip(const float* __restrict__ inmap, float* __restrict__ outmap,
                                 unsigned int stride, unsigned int axislength, unsigned int slides) {{

                unsigned int i = {Defines.IndexX}, j = {Defines.IndexY}, k = {Defines.IndexZ};
                if (i >= stride || j >= axislength || k >= slides) {{
                    return;
                }}

                unsigned int offset = i + stride * axislength * k;
                unsigned int inmap_idx = offset + stride * j;
                unsigned int outmap_idx = offset + stride * (axislength - j - 1);

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "flip");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;

            uint stride = (args[2] as uint?).Value;
            uint axislength = (args[3] as uint?).Value;
            uint slides = (args[4] as uint?).Value;

            for (uint s = 0; s < slides; s += SlidesPerExecute) {
                uint sl = Math.Min(SlidesPerExecute, slides - s);

                Kernel.Execute(
                    indexes: (stride, axislength, sl),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(s * stride * axislength),
                    outmap.ElementPtr(s * stride * axislength),
                    stride, axislength, sl
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint stride) || stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            if (!(args[3] is uint axislength) || axislength < 1) {
                throw new ArgumentException(nameof(axislength));
            }

            if (!(args[4] is uint slides) || slides < 0) {
                throw new ArgumentException(nameof(slides));
            }

            uint length = stride * axislength * slides;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < length) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < length) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
