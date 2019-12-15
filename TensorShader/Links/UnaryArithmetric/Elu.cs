using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>Elu</summary>
        public static Field Elu(Field x, float alpha = 1) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Elu(x, y, alpha);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>Elu</summary>
    internal class Elu : UnaryArithmetric {
        /// <summary>Alpha</summary>
        public float Alpha { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Elu(Field infield, Field outfield, float alpha)
            : base(infield, outfield) {
            this.Alpha = alpha;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Elu(X.Value, Alpha));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(EluGrad(Alpha, Y.Grad, X.Value));
            }
        }
    }
}
