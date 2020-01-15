namespace TensorShaderCudaBackend {

    /// <summary>上限値</summary>
    public static class Limits {

        /// <summary>要素数</summary>
        public static ulong Length => CudaArray<float>.MaxLength;

        /// <summary>チャネル数(4096)</summary>
        public static uint Channels => 0x1000;

        /// <summary>横幅(8192)</summary>
        public static uint Width => 0x2000;

        /// <summary>縦幅(8192)</summary>
        public static uint Height => 0x2000;

        /// <summary>奥行き(8192)</summary>
        public static uint Depth => 0x2000;

        /// <summary>バッチ数(32768‬)</summary>
        public static uint Batches => 0x8000;

        /// <summary>カーネルサイズ(512)</summary>
        public static uint KernelSize => 512;

        /// <summary>チャネル数チェック</summary>
        public static bool CheckChannels(params uint[] channels) {
            foreach (uint ch in channels) {
                if (ch < 1 || ch > Channels) {
                    return false;
                }
            }

            return true;
        }

        /// <summary>横幅チェック</summary>
        public static bool CheckWidth(uint width, uint min_width = 1) {
            return width >= min_width && width <= Width;
        }

        /// <summary>縦幅チェック</summary>
        public static bool CheckHeight(uint height, uint min_height = 1) {
            return height >= min_height && height <= Height;
        }

        /// <summary>奥行きチェック</summary>
        public static bool CheckDepth(uint depth, uint min_depth = 1) {
            return depth >= min_depth && depth <= Depth;
        }

        /// <summary>バッチ数チェック</summary>
        public static bool CheckBatches(uint batches) {
            return batches >= 1 && batches <= Batches;
        }

        /// <summary>カーネルサイズチェック</summary>
        public static bool CheckKernelSize(params uint[] ksizes) {
            uint prod = 1;

            foreach (uint ksize in ksizes) {
                if ((ksize & 1) != 1) {
                    return false;
                }

                prod *= ksize;
            }

            if (prod > KernelSize) {
                return false;
            }

            return true;
        }

        /// <summary>倍数チェック</summary>
        public static bool CheckMultipleNum(uint multiple, params uint[] nums) {
            foreach (uint num in nums) {
                if ((num % multiple) != 0) {
                    return false;
                }
            }

            return true;
        }
    }
}
