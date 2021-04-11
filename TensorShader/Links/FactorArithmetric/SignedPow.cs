using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>符号付きべき関数</summary>
        public static Field SignedPow(Field x, Field alpha) {
            Field y = new();
            Link link = new Links.FactorArithmetric.SignedPow(x, alpha, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.FactorArithmetric {
    /// <summary>符号付きべき関数</summary>
    public class SignedPow : FactorArithmetric {
        /// <summary>コンストラクタ</summary>
        public SignedPow(Field infield, Field factorfield, Field outfield)
            : base(infield, factorfield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(SignedPow(X.Value, Factor.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Factor.Value * Pow(Abs(X.Value), Factor.Value - 1) * Y.Grad);
            }

            if (Factor.EnableBackprop) {
                Factor.AddGrad(Sum(NanAsZero(Y.Grad * Log(Abs(X.Value)) * Y.Value)));
            }
        }
    }
}
