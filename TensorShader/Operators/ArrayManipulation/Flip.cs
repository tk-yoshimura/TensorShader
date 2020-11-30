using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>軸反転</summary>
    internal class Flip : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>軸長さ</summary>
        public int Length { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Flip(Shape shape, int axis) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
            this.Stride = shape.Stride(axis);
            this.Length = shape[axis];
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor intensor = tensors[0], outtensor = tensors[1];

            TensorShaderCudaBackend.ArrayManipulation.Flip((uint)Stride, (uint)Length, (uint)(outtensor.Length / (Stride * Length)), intensor.Buffer, outtensor.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor intensor, Tensor outtensor) {
            Execute(new Tensor[] { intensor, outtensor });
        }
    }
}
