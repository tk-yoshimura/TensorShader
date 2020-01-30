using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>パターンコピー</summary>
    public sealed class PatternCopy : Shader {

        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>実行あたりのスライド数</summary>
        public static uint SlidesPerExecute => 0x8000;

        /// <summary>コンストラクタ</summary>
        public PatternCopy() {
            string code = $@"

            __global__ void pattern_copy(const float* __restrict__ inmap, float* __restrict__ outmap,
                                         unsigned int inmap_stride, unsigned int outmap_stride,
                                         unsigned int copy_length, unsigned int slides) {{

                unsigned int i = {Defines.IndexX}, j = {Defines.IndexY};
                if (i >= copy_length || j >= slides) {{
                    return;
                }}

                unsigned int inmap_idx = i + j * inmap_stride;
                unsigned int outmap_idx = i + j * outmap_stride;

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "pattern_copy");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint inmap_offset = (args[2] as uint?).Value;
            uint inmap_index = (args[3] as uint?).Value;
            uint outmap_offset = (args[4] as uint?).Value;
            uint outmap_index = (args[5] as uint?).Value;
            uint inmap_stride = (args[6] as uint?).Value;
            uint outmap_stride = (args[7] as uint?).Value;
            uint copylength = (args[8] as uint?).Value;
            uint slides = (args[9] as uint?).Value;

            for (uint s = 0; s < slides; s += SlidesPerExecute) {
                uint sl = Math.Min(SlidesPerExecute, slides - s);

                Kernel.Execute(
                    indexes: (copylength, sl),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(inmap_offset + inmap_index + s * inmap_stride),
                    outmap.ElementPtr(outmap_offset + outmap_index + s * outmap_stride),
                    inmap_stride, outmap_stride, copylength, sl
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 10) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint inmap_offset)) {
                throw new ArgumentException(nameof(inmap_offset));
            }

            if (!(args[4] is uint outmap_offset)) {
                throw new ArgumentException(nameof(outmap_offset));
            }

            if (!(args[8] is uint copy_length) || copy_length < 1) {
                throw new ArgumentException(nameof(copy_length));
            }

            if (!(args[6] is uint inmap_stride) || inmap_stride < copy_length) {
                throw new ArgumentException(nameof(inmap_stride));
            }

            if (!(args[7] is uint outmap_stride) || outmap_stride < copy_length) {
                throw new ArgumentException(nameof(outmap_stride));
            }

            if (!(args[3] is uint inmap_index) || inmap_index + copy_length > inmap_stride) {
                throw new ArgumentException(nameof(inmap_index));
            }

            if (!(args[5] is uint outmap_index) || outmap_index + copy_length > outmap_stride) {
                throw new ArgumentException(nameof(outmap_index));
            }

            if (!(args[9] is uint slides)) {
                throw new ArgumentException(nameof(slides));
            }

            uint inmap_length = inmap_offset + inmap_stride * slides;
            uint outmap_length = outmap_offset + outmap_stride * slides;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < inmap_length) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < outmap_length) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
