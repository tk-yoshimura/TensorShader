using TensorShader;
using TensorShader.Links.UnaryArithmetric;

using static TensorShader.VariableNode;

namespace CustomLink {

    public static class CustomUnaryArithmetric {
        /// <summary>カスタム1項演算</summary>
        public static Field ExpSin(Field x) {
            Field y = new Field();
            Link link = new ExpSin(x, y);

            link.Forward();

            return y;
        }
    }

    /// <summary>カスタム1項演算</summary>
    internal class ExpSin : UnaryArithmetric {

        /// <summary>コンストラクタ</summary>
        public ExpSin(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(UnaryArithmetric(X.Value, "expsin_ew", "#y = expf(sinf(#x));"));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                VariableNode dx = UnaryArithmetric(X.Value, "dexpsin_ew", "#y = expf(sinf(#x)) * cosf(#x);");

                X.AddGrad(Y.Grad * dx);
            }
        }
    }
}
