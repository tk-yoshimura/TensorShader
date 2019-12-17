using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>最大値</summary>
        public static Field Maximum(Field x, float c) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Maximum(x, y, c);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>最大値</summary>
    internal class Maximum : UnaryArithmetric {
        /// <summary>定数項</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Maximum(Field infield, Field outfield, float c)
            : base(infield, outfield) {

            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Maximum(X.Value, Constant));
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
