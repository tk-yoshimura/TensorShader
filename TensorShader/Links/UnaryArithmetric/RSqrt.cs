using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>逆平方根</summary>
        public static Field Rsqrt(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Rsqrt(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>逆平方根</summary>
    internal class Rsqrt : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Rsqrt(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Rsqrt(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(-0.5f * Cube(Y.Value) * Y.Grad);
            }
        }
    }
}
