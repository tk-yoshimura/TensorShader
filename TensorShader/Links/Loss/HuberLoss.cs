using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>HuberLoss</summary>
        public static Field HuberLoss(Field x, Field t, float delta) {
            Field y = new Field();
            Link link = new Links.Loss.HuberLoss(x, t, y, delta);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Loss {
    /// <summary>HuberLoss</summary>
    public class HuberLoss : Link {
        private VariableNode d, step_d, flip_d;

        /// <summary>L1-2境界</summary>
        public float Delta { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>教師信号項</summary>
        protected Field T => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public HuberLoss(Field xfield, Field tfield, Field outfield, float delta)
            : base(new Field[] { xfield, tfield }, outfield) {
            this.Delta = delta;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            d = X.Value - T.Value;

            VariableNode abs_d = Abs(d);
            VariableNode squa_d = Square(d);

            step_d = GreaterThan(abs_d, Delta);
            flip_d = 1 - step_d;

            Y.AssignValue(step_d * (abs_d - Delta / 2) * Delta + flip_d * squa_d / 2);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                VariableNode dx = step_d * Sign(d) * Delta + flip_d * d;

                X.AddGrad(dx * Y.Grad);
            }
        }
    }
}
