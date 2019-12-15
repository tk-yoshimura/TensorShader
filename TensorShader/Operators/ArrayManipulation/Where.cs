using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>条件選択</summary>
    internal class Where : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Where(Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.In, shape),
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor incondition = tensors[0], intensor1 = tensors[1], intensor2 = tensors[2], outtensor = tensors[3];

            TensorShaderCudaBackend.Elementwise.Lerp((uint)incondition.Length, incondition.Buffer, intensor1.Buffer, intensor2.Buffer, outtensor.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor incondition, Tensor intensor1, Tensor intensor2, Tensor outtensor) {
            Execute(new Tensor[] { incondition, intensor1, intensor2, outtensor });
        }
    }
}
