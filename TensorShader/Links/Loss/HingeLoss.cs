using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>HingeLoss</summary>
        public static Field HingeLoss(Field x, Field t) {
            Field y = new();
            Link link = new Links.Loss.HingeLoss(x, t, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Loss {
    /// <summary>HingeLoss</summary>
    public class HingeLoss : Link {
        private VariableNode xt;

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>教師信号項</summary>
        protected Field T => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public HingeLoss(Field xfield, Field tfield, Field outfield)
            : base(new Field[] { xfield, tfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            xt = 1 - X.Value * T.Value;

            Y.AssignValue(Maximum(0, xt));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Step(xt) * -T.Value * Y.Grad);
            }
        }
    }
}
