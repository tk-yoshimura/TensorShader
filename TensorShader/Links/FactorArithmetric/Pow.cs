using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>べき関数</summary>
        public static Field Pow(Field x, Field alpha) {
            Field y = new();
            Link link = new Links.FactorArithmetric.Pow(x, alpha, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.FactorArithmetric {
    /// <summary>べき関数</summary>
    public class Pow : FactorArithmetric {
        /// <summary>コンストラクタ</summary>
        public Pow(Field infield, Field factorfield, Field outfield)
            : base(infield, factorfield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Pow(X.Value, Factor.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Factor.Value * Pow(X.Value, Factor.Value - 1) * Y.Grad);
            }

            if (Factor.EnableBackprop) {
                Factor.AddGrad(Sum(NanAsZero(Y.Grad * Log(X.Value) * Y.Value)));
            }
        }
    }
}
