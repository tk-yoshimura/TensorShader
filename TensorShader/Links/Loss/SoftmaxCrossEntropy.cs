using System;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>SoftmaxCrossEntropy</summary>
        public static Field SoftmaxCrossEntropy(Field x, Field t) {
            Field y = new Field();
            Link link = new Links.Loss.SoftmaxCrossEntropy(x, t, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Loss {
    /// <summary>SoftmaxCrossEntropy</summary>
    public class SoftmaxCrossEntropy : Link {
        private VariableNode x_softmax;

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>教師信号項</summary>
        protected Field T => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public SoftmaxCrossEntropy(Field xfield, Field tfield, Field outfield)
            : base(new Field[] { xfield, tfield }, outfield) {
            if (xfield.Shape.Type != ShapeType.Map || xfield.Shape.Ndim < 2 || xfield.Shape != tfield.Shape) {
                throw new ArgumentException($"{nameof(xfield)}, {nameof(tfield)}");
            }
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            x_softmax = Softmax(X.Value);

            Y.AssignValue(-T.Value * Log(x_softmax + 1e-5f));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad((x_softmax - T.Value) * Y.Grad);
            }
        }
    }
}
