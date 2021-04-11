using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素数正規化</summary>
        public static Field ComplexNormalize(Field x) {
            Field y = new();
            Link link = new Links.ComplexArithmetric.ComplexNormalize(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>複素数正規化</summary>
    public class ComplexNormalize : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexNormalize(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexNormalize(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexNormalizeGrad(Y.Grad, X.Value));
            }
        }
    }
}
