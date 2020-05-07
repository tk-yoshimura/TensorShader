using System.Linq;

namespace TensorShaderUtil.Iterator {
    /// <summary>バッチインデクサを同順で繰り返すイテレータ</summary>
    public class RepeatIterator : Iterator {
        private int pos;

        /// <summary>コンストラクタ</summary>
        public RepeatIterator(int num_batches, int counts)
            : base(num_batches, counts) {
            this.pos = 0;
        }

        /// <summary>次のインデクサ</summary>
        public override int[] Next() {
            IncreaseIteration();

            if (pos + NumBatches >= Counts) {
                IncreaseEpoch();
            }

            int[] batch_indexes = (new int[NumBatches]).Select((_, idx) => (pos + idx) % Counts).ToArray();

            pos = (pos + NumBatches) % Counts;

            return batch_indexes;
        }

        /// <summary>Iterationをスキップする</summary>
        /// <remarks>増加時イベントは生じない</remarks>
        public override void SkipIteration(long iter) {
            base.SkipIteration(iter);

            pos = (int)((pos + NumBatches * iter) % Counts);
        }
    }
}
