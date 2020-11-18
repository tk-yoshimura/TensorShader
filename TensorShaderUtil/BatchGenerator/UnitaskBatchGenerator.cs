using System;
using TensorShader;

namespace TensorShaderUtil.BatchGenerator {
    /// <summary>ユニタスクバッチ生成</summary>
    public abstract class UnitaskBatchGenerator : BatchGenerator {
        /// <summary>コンストラクタ</summary>
        public UnitaskBatchGenerator(Shape data_shape, int num_batches)
            : base(data_shape, num_batches) {
            this.Requested = false;
        }

        /// <summary>データ生成をリクエスト</summary>
        /// <param name="indexes">バッチのデータインデクス</param>
        /// <remarks>indexesをnullにした場合、GenerateDataのindexを0として呼び出される</remarks>
        public override sealed void Request(int[] indexes = null) {
            if (indexes != null && indexes.Length != NumBatches) {
                throw new ArgumentException(nameof(indexes));
            }

            for (int i = 0; i < NumBatches; i++) {
                int index = indexes != null ? indexes[i] : 0;

                GenerateToInsert(this, i, index);
            }

            Requested = true;
        }

        /// <summary>バッチを受け取る</summary>
        public override sealed NdimArray<float> Receive() {
            if (!Requested) {
                throw new InvalidOperationException();
            }

            return Value;
        }
    }
}
