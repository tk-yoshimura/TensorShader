using System.Collections.Generic;

namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>定数2項演算</summary>
    internal abstract class BinaryRightConstantArithmetric : Operator {
        /// <summary>マップ形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>定数項</summary>
        protected readonly float Constant;

        /// <summary>コンストラクタ</summary>
        public BinaryRightConstantArithmetric(float c, Shape shape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Constant = c;
            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
