using System.Collections.Generic;

namespace TensorShader.Operators.TrinaryArithmetric {
    /// <summary>定数3項演算</summary>
    internal abstract class TrinaryBiConstantArithmetric : Operator {
        /// <summary>マップ形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>定数項1</summary>
        protected readonly float Constant1;

        /// <summary>定数項2</summary>
        protected readonly float Constant2;

        /// <summary>コンストラクタ</summary>
        public TrinaryBiConstantArithmetric(float c1, float c2, Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Constant1 = c1;
            this.Constant2 = c2;
            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor inmap3, Tensor outmap) {
            Execute(new Tensor[] { inmap1, inmap2, inmap3, outmap });
        }
    }
}
