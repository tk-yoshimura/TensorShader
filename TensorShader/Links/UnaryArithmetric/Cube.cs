using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3乗</summary>
        public static Field Cube(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Cube(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>3乗</summary>
    internal class Cube : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Cube(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Cube(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(3.0f * Square(X.Value) * Y.Grad);
            }
        }
    }
}
