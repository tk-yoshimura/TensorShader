using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>インデクス</summary>
    internal class Index : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>ソート長さ</summary>
        public int Length { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Index(Shape shape, int axis) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
            this.Stride = shape.Stride(axis);
            this.Length = shape[axis];
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor reftensor = tensors[0];

            TensorShaderCudaBackend.Indexer.Index((uint)Stride, (uint)Length, (uint)(reftensor.Length / (Stride * Length)), reftensor.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor reftensor) {
            Execute(new Tensor[] { reftensor });
        }
    }
}
