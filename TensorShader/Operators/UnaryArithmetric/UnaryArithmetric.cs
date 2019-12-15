using System.Collections.Generic;

namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>1項演算</summary>
    internal abstract class UnaryArithmetric : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public UnaryArithmetric(Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
