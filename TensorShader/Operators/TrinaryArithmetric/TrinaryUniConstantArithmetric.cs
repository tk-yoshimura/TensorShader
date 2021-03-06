using System.Collections.Generic;

namespace TensorShader.Operators.TrinaryArithmetric {
    /// <summary>定数3項演算</summary>
    internal abstract class TrinaryUniConstantArithmetric : Operator {
        /// <summary>マップ形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>定数項</summary>
        protected readonly float Constant;

        /// <summary>コンストラクタ</summary>
        public TrinaryUniConstantArithmetric(float c, Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Constant = c;
            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor inmap3, Tensor outmap) {
            Execute(new Tensor[] { inmap1, inmap2, inmap3, outmap });
        }
    }
}
