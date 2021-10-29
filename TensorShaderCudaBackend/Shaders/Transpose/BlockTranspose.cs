using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Transpose {
    /// <summary>ブロックごとに転置</summary>
    public class BlockTranspose : Shader {

        /// <summary>ブロック</summary>
        public uint Block { private set; get; }

        /// <summary>転置N</summary>
        public uint N { private set; get; }

        /// <summary>転置M</summary>
        public uint M { private set; get; }

        /// <summary>実行あたりのNM数(2^15=32768‬)</summary>
        public static uint PointsPerNMCounts => 0x8000;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Block)} = {Block} {nameof(N)} = {N} {nameof(M)} = {M}";

        /// <summary>コンストラクタ</summary>
        public BlockTranspose(uint block, uint n, uint m) {
            if (!Limits.CheckChannels(block)) {
                throw new ArgumentException(nameof(block));
            }

            this.Block = block;
            this.N = n;
            this.M = m;

            string code = $@"

            __global__ void block_transpose(const float* __restrict__ inmap, float* __restrict__ outmap, unsigned int offset) {{

                unsigned int i = {Defines.IndexX}, j = offset + {Defines.IndexY};

                if (i >= {Block} || j >= {N * M}) {{
                    return;
                }}

                unsigned int n = j % {N}, m = j / {N};

                unsigned int inmap_idx = i + {Block} * (n + m * {N});
                unsigned int outmap_idx = i + {Block} * (m + n * {M});

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "block_transpose");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;

            for (uint offset = 0, nm = N * M; offset < nm; offset += PointsPerNMCounts) {
                uint nml = Math.Min(PointsPerNMCounts, nm - offset);

                Kernel.Execute(
                    indexes: (Block, nml),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap,
                    outmap,
                    offset
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 2) {
                throw new ArgumentException(nameof(args));
            }

            uint length = Block * N * M;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < length) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < length) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
