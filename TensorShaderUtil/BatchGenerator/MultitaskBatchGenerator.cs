using System;
using System.Linq;
using System.Threading.Tasks;
using TensorShader;

namespace TensorShaderUtil.BatchGenerator {
    /// <summary>マルチタスクバッチ生成</summary>
    public abstract class MultitaskBatchGenerator : BatchGenerator {
        private Task[] tasks;

        /// <summary>コンストラクタ</summary>
        public MultitaskBatchGenerator(Shape data_shape, int num_batches)
            : base(data_shape, num_batches) {
            this.Requested = false;
        }

        /// <summary>データ生成をリクエスト</summary>
        /// <param name="indexes">バッチのデータインデクス</param>
        /// <remarks>indexesをnullにした場合、GenerateDataのindexを0として呼び出される</remarks>
        public override sealed void Request(int[] indexes = null) {
            if (indexes is not null && indexes.Length != NumBatches) {
                throw new ArgumentException(null, nameof(indexes));
            }

            if (tasks is not null) {
                Task.WaitAll(tasks);
            }

            tasks = Enumerable.Range(0, NumBatches)
                .Select(
                    (i) => Task.Run(() => GenerateToInsert(this, i, indexes is not null ? indexes[i] : 0))
                ).ToArray();

            Requested = true;
        }

        /// <summary>バッチを受け取る</summary>
        public override sealed NdimArray<float> Receive() {
            if (!Requested || tasks is null) {
                throw new InvalidOperationException();
            }

            Task.WaitAll(tasks);

            return Value;
        }
    }
}
