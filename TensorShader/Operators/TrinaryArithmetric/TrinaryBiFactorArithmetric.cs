using System.Collections.Generic;

namespace TensorShader.Operators.TrinaryArithmetric {
    /// <summary>指数3項演算</summary>
    internal abstract class TrinaryBiFactorArithmetric : Operator {
        /// <summary>マップ形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrinaryBiFactorArithmetric(Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Scalar),
                (ArgumentType.In, Shape.Scalar),
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor inmap3, Tensor outmap) {
            Execute(new Tensor[] { inmap1, inmap2, inmap3, outmap });
        }
    }
}
