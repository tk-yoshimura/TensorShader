using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>符号付きべき関数</summary>
        public static Field SignedPow(Field x, float alpha) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.SignedPow(x, y, alpha);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>符号付きべき関数</summary>
    internal class SignedPow : UnaryArithmetric {
        /// <summary>指数</summary>
        public float Alpha { private set; get; }

        /// <summary>コンストラクタ</summary>
        public SignedPow(Field infield, Field outfield, float alpha)
            : base(infield, outfield) {

            this.Alpha = alpha;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(SignedPow(X.Value, Alpha));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Alpha * Pow(Abs(X.Value), Alpha - 1) * Y.Grad);
            }
        }
    }
}
