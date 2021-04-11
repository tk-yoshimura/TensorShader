using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素共役</summary>
        public static Field ComplexConjugate(Field x) {
            Field y = new();
            Link link = new Links.ComplexArithmetric.ComplexConjugate(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>複素共役</summary>
    public class ComplexConjugate : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexConjugate(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexConjugate(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexConjugate(Y.Grad));
            }
        }
    }
}
