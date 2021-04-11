using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>2乗誤差</summary>
        public static Field SquareError(Field x, Field t) {
            Field y = new();
            Link link = new Links.Loss.SquareError(x, t, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Loss {
    /// <summary>2乗誤差</summary>
    public class SquareError : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>教師信号項</summary>
        protected Field T => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public SquareError(Field xfield, Field tfield, Field outfield)
            : base(new Field[] { xfield, tfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Square(X.Value - T.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(2 * (X.Value - T.Value) * Y.Grad);
            }
        }
    }
}
