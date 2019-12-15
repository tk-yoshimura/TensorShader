using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>最小値</summary>
        public static Field Minimum(Field x, float c) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Minimum(x, y, c);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>最小値</summary>
    internal class Minimum : UnaryArithmetric {
        /// <summary>定数項</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Minimum(Field infield, Field outfield, float c)
            : base(infield, outfield) {
            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Minimum(X.Value, Constant));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad * Equal(X.Value, Y.Value));
            }
        }
    }
}
