using System;
using System.Collections.ObjectModel;

namespace TensorShaderUtil.DataSplitUtil {

    /// <summary>パッチワーク座標</summary>
    /// <remarks>ワークスペースを節約するためデータを分割・統合する際に構成情報として用いる</remarks>
    public class PatchworkCoord {
        private readonly int[] block_coords, block_sizes, patch_coords, patch_sizes;

        /// <summary>長さ</summary>
        public int Length { private set; get; }

        /// <summary>ブロックサイズ</summary>
        public int BlockSize { private set; get; }

        /// <summary>パッチサイズ</summary>
        public int PatchSize { private set; get; }

        /// <summary>マージン</summary>
        public int Margin { private set; get; }

        /// <summary>ブロック数</summary>
        public int Blocks { private set; get; }

        /// <summary>ブロック座標</summary>
        public ReadOnlyCollection<int> BlockCoords => Array.AsReadOnly(block_coords);

        /// <summary>ブロックサイズ</summary>
        public ReadOnlyCollection<int> BlockSizes => Array.AsReadOnly(block_sizes);

        /// <summary>パッチ座標</summary>
        public ReadOnlyCollection<int> PatchCoords => Array.AsReadOnly(patch_coords);

        /// <summary>パッチサイズ</summary>
        public ReadOnlyCollection<int> PatchSizes => Array.AsReadOnly(patch_sizes);

        /// <summary>コンストラクタ</summary>
        public PatchworkCoord(int length, int blocksize, int margin) {
            if (length < 1 || margin < 0 || (blocksize <= 2 * margin && length > blocksize)) {
                throw new ArgumentException();
            }

            this.Length = length;

            if (length <= blocksize) {
                this.BlockSize = blocksize;
                this.PatchSize = blocksize;
                this.Margin = margin;
                this.Blocks = 1;

                this.block_coords = this.patch_coords = new int[] { 0 };
                this.block_sizes = this.patch_sizes = new int[] { length };
            }
            else {
                this.BlockSize = blocksize;
                this.PatchSize = blocksize - 2 * margin;
                this.Margin = margin;
                this.Blocks = (length - 2 * margin + PatchSize - 1) / PatchSize;

                this.block_coords = new int[Blocks];
                this.block_sizes = new int[Blocks];
                this.patch_coords = new int[Blocks];
                this.patch_sizes = new int[Blocks];

                patch_sizes[0] = PatchSize + margin;
                block_sizes[0] = BlockSize;
                patch_coords[0] = block_coords[0] = 0;

                for (int i = 1; i < Blocks - 1; i++) {
                    patch_sizes[i] = PatchSize;
                    block_sizes[i] = BlockSize;
                    patch_coords[i] = i * PatchSize + margin;
                    block_coords[i] = i * PatchSize;
                }

                patch_sizes[Blocks - 1] = PatchSize + margin;
                block_sizes[Blocks - 1] = BlockSize;
                patch_coords[Blocks - 1] = length - patch_sizes[Blocks - 1];
                block_coords[Blocks - 1] = length - block_sizes[Blocks - 1];
            }
        }
    }
}
