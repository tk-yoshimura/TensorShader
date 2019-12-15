using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>絶対誤差</summary>
        public static Field AbsoluteError(Field x, Field t) {
            Field y = new Field();
            Link link = new Links.Loss.AbsoluteError(x, t, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Loss {
    /// <summary>絶対誤差</summary>
    public class AbsoluteError : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>教師信号項</summary>
        protected Field T => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public AbsoluteError(Field xfield, Field tfield, Field outfield)
            : base(new Field[] { xfield, tfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Abs(X.Value - T.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Sign(X.Value - T.Value) * Y.Grad);
            }
        }
    }
}
