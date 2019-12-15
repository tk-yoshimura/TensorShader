using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>軸反転</summary>
        public static Field Flip(Field x, int axis) {
            Field y = new Field();

            Link link = new Links.ArrayManipulation.Flip(x, y, axis);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>軸反転</summary>
    public class Flip : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>軸</summary>
        protected readonly int Axis;

        /// <summary>コンストラクタ</summary>
        public Flip(Field x, Field y, int axis)
            : base(new Field[] { x }, y) {
            this.Axis = axis;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Flip(X.Value, Axis));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Flip(Y.Grad, Axis));
            }
        }
    }
}
