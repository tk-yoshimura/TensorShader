using System;
using System.Linq;
using TensorShader;

namespace TensorShaderUtil.BatchGenerator {
    /// <summary>バッチ生成</summary>
    public abstract class BatchGenerator : IBatchGenerator {
        /// <summary>生成データ配列</summary>
        protected NdimArray<float> Value { private set; get; }

        /// <summary>データ形状</summary>
        public Shape DataShape { get; }

        /// <summary>バッチ形状</summary>
        public Shape BatchShape { get; }

        /// <summary>バッチ数</summary>
        public int NumBatches { get; }

        /// <summary>リクエスト済</summary>
        public bool Requested { protected set; get; }

        /// <summary>コンストラクタ</summary>
        public BatchGenerator(Shape data_shape, int num_batches) {
            if (data_shape is null) {
                throw new ArgumentNullException(nameof(data_shape));
            }
            if (data_shape.Batch > 1) {
                throw new ArgumentException(null, nameof(data_shape));
            }
            if (!(new ShapeType[] { ShapeType.Scalar, ShapeType.Vector, ShapeType.Map }.Contains(data_shape.Type))) {
                throw new ArgumentException(null, nameof(data_shape));
            }
            if (num_batches < 1) {
                throw new ArgumentException(null, nameof(num_batches));
            }

            this.DataShape = data_shape;

            switch (data_shape.Type) {
                case ShapeType.Scalar:
                    this.BatchShape = Shape.Vector(num_batches);
                    break;
                case ShapeType.Vector:
                    this.BatchShape = Shape.Map0D(data_shape.Channels, num_batches);
                    break;
                case ShapeType.Map:
                    int[] s = data_shape;
                    s[s.Length - 1] = num_batches;

                    this.BatchShape = new Shape(ShapeType.Map, s);
                    break;
            }

            this.Value = BatchShape;
            this.NumBatches = num_batches;
        }

        /// <summary>データ生成</summary>
        /// <param name="index">データインデクス</param>
        public abstract NdimArray<float> GenerateData(int index);

        /// <summary>データ生成をリクエスト</summary>
        /// <param name="indexes">バッチのデータインデクス</param>
        public abstract void Request(int[] indexes = null);

        /// <summary>バッチを受け取る</summary>
        public abstract NdimArray<float> Receive();

        /// <summary>バッチを取得する</summary>
        public NdimArray<float> Get(int[] indexes = null) {
            Request(indexes);
            return Receive();
        }

        /// <summary>データ生成及びデータ挿入</summary>
        protected Action<BatchGenerator, int, int> GenerateToInsert
            = (BatchGenerator generator, int i, int index) => {
                NdimArray<float> data = generator.GenerateData(index);

                if (data.Shape.DataShape != generator.DataShape) {
                    throw new Exception("Invalid data length.");
                }

                Buffer.BlockCopy(data.Value, 0, generator.Value.Value, i * data.Length * sizeof(float), data.Length * sizeof(float));
            };
    }
}
