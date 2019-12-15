using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>立方根</summary>
        public static Field Cbrt(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Cbrt(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>立方根</summary>
    internal class Cbrt : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Cbrt(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Cbrt(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Rcp(3.0f * Square(Y.Value)) * Y.Grad);
            }
        }
    }
}
