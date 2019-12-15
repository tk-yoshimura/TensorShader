using System;

namespace TensorShaderUtil.Iterator {
    /// <summary>イテレータ</summary>
    public abstract class Iterator {
        /// <summary>Epoch</summary>
        public long Epoch { protected set; get; }

        /// <summary>Iteration</summary>
        public long Iteration { protected set; get; }

        /// <summary>バッチ数</summary>
        public int NumBatches { private set; get; }

        /// <summary>データ数</summary>
        public int Counts { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Iterator(int num_batches, int counts) {
            if (num_batches > counts || num_batches <= 0) {
                throw new ArgumentException($"{nameof(num_batches)}, {nameof(counts)}");
            }

            this.Epoch = 0;
            this.Iteration = 0;

            this.NumBatches = num_batches;
            this.Counts = counts;
        }

        /// <summary>次のインデクサ</summary>
        public abstract int[] Next();
    }
}
