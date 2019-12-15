using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>条件選択</summary>
        public static Field Where(Field condition, Field x1, Field x2) {
            Field y = new Field();
            Link link = new Links.ArrayManipulation.Where(condition, x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>条件選択</summary>
    public class Where : Link {
        /// <summary>条件項</summary>
        protected Field Condition => InFields[0];

        /// <summary>入力項1</summary>
        protected Field X1 => InFields[1];

        /// <summary>入力項2</summary>
        protected Field X2 => InFields[2];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>形状</summary>
        protected Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Where(Field condition, Field x1, Field x2, Field y)
            : base(new Field[] { condition, x1, x2 }, y) {
            this.Shape = condition.Shape;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Where(Condition.Value, X1.Value, X2.Value, Shape));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(Condition.Value * Y.Grad);
            }

            if (X2.EnableBackprop) {
                X2.AddGrad((1 - Condition.Value) * Y.Grad);
            }
        }
    }
}
