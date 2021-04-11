using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>2乗</summary>
        public static Field Square(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Square(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>2乗</summary>
    internal class Square : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Square(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Square(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(2.0f * X.Value * Y.Grad);
            }
        }
    }
}
