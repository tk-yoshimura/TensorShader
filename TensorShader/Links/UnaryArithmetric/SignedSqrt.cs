using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>符号付き平方根</summary>
        public static Field SignedSqrt(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.SignedSqrt(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>符号付き平方根</summary>
    internal class SignedSqrt : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public SignedSqrt(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(SignedSqrt(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Rcp(2.0f * Sign(X.Value) * Y.Value) * Y.Grad);
            }
        }
    }
}
