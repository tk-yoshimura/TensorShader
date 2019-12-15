using System.Collections.Generic;

namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>2項演算</summary>
    internal abstract class BinaryArithmetric : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public BinaryArithmetric(Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outmap) {
            Execute(new Tensor[] { inmap1, inmap2, outmap });
        }
    }
}
