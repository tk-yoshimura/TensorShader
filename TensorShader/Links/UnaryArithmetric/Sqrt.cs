using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>平方根</summary>
        public static Field Sqrt(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Sqrt(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>平方根</summary>
    internal class Sqrt : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sqrt(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Sqrt(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Rcp(2.0f * Y.Value) * Y.Grad);
            }
        }
    }
}
