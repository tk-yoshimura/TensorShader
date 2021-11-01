using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.ArrayManipulation {

    /// <summary>領域加算</summary>
    public sealed class RegionAdd : Shader {
        /// <summary>識別子</summary>
        public override sealed string Signature => $"{GetType().Name.Split(',').Last()}";

        /// <summary>コンストラクタ</summary>
        public RegionAdd() {
            string code = $@"

            __global__ void region_add(const float* __restrict__ inmap, float* __restrict__ outmap, unsigned int n) {{

                unsigned int i = {Defines.IndexX};

                if(i >= n){{
                    return;
                }}

                outmap[i] += inmap[i];
            }}";

            this.Kernel = new Kernel(code, "region_add");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint n = (args[2] as uint?).Value;
            uint src_index = (args[3] as uint?).Value;
            uint dst_index = (args[4] as uint?).Value;

            Kernel.Execute(
                n, dynamic_shared_memory_bytes: 0, stream,
                inmap.ElementPtr(src_index),
                outmap.ElementPtr(dst_index),
                n
            );
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 5) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[2] is not uint n || n < 1) {
                throw new ArgumentException(nameof(n));
            }

            if (args[3] is not uint src_index) {
                throw new ArgumentException(nameof(src_index));
            }

            if (args[4] is not uint dst_index) {
                throw new ArgumentException(nameof(dst_index));
            }

            if (args[0] is not CudaArray<float> inmap || inmap.Length < src_index + n) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < dst_index + n) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
