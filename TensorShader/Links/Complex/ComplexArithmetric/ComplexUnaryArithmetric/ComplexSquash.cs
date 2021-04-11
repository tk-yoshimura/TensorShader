using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素数に1/(1+sqrt(norm))を乗ずる</summary>
        public static Field ComplexSquash(Field x) {
            Field y = new();
            Link link = new Links.ComplexArithmetric.ComplexSquash(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>複素数に1/(1+sqrt(norm))を乗ずる</summary>
    public class ComplexSquash : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexSquash(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexSquash(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(NanAsZero(ComplexSquashGrad(Y.Grad, X.Value)));
            }
        }
    }
}
