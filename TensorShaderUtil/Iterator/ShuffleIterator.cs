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
            : base(num_batches, counts){
            this.pos = 0;
            this.indexes = (new int[counts]).Select((_, idx) => idx).ToArray();
            this.random = random;
            Shuffle(indexes, random);
        }

        /// <summary>次のインデクサ</summary>
        public override int[] Next() {
            if (pos + NumBatches > Counts) {
                Shuffle(indexes, random);
                pos = 0;
                Epoch++;
            }

            int[] batch_indexes = new int[NumBatches];
            Array.Copy(indexes, pos, batch_indexes, 0, NumBatches);

            pos += NumBatches;
            Iteration++;

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
    }
}
