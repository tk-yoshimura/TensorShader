using System;
using System.Linq;

namespace TensorShaderUtil.Iterator {
    /// <summary>バッチインデクサをランダムに生成するイテレータ</summary>
    public class ShuffleIterator : Iterator {
        private int pos;
        private int[] indexes;
        private readonly Random random;

        /// <summary>コンストラクタ</summary>
        public ShuffleIterator(int num_batches, int counts, Random random)
            : base(num_batches, counts) {
            this.pos = 0;
            this.indexes = (new int[counts]).Select((_, idx) => idx).ToArray();
            this.random = random;
            Shuffle(indexes, random);
        }

        /// <summary>次のインデクサ</summary>
        public override int[] Next() {
            IncreaseIteration();

            int[] batch_indexes = new int[NumBatches];

            if (pos + NumBatches >= Counts) {
                Array.Copy(indexes, pos, batch_indexes, 0, Counts - pos);
                Shuffle(indexes, random);
                Array.Copy(indexes, 0, batch_indexes, Counts - pos, NumBatches - Counts + pos);

                IncreaseEpoch();
            }
            else {
                Array.Copy(indexes, pos, batch_indexes, 0, NumBatches);
            }

            pos = (pos + NumBatches) % Counts;

            return batch_indexes;
        }

        /// <summary>シャッフル</summary>
        protected static void Shuffle(int[] array, Random random) {
            int i = array.Length;
            while (i > 1) {
                i--;

                int j = random.Next(i + 1);
                int swap = array[j]; array[j] = array[i]; array[i] = swap;
            }
        }

        /// <summary>Iterationをスキップする</summary>
        /// <remarks>増加時イベントは生じない</remarks>
        public override void SkipIteration(long iter) {
            long prev_epoch = Epoch;

            base.SkipIteration(iter);

            long post_epoch = Epoch;

            pos = (int)((pos + NumBatches * iter) % Counts);

            for (long epoch = prev_epoch; epoch < post_epoch; epoch++) {
                Shuffle(indexes, random);
            }
        }
    }
}
