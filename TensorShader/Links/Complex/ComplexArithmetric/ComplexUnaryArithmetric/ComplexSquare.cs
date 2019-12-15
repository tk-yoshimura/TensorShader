using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素2乗</summary>
        public static Field ComplexSquare(Field x) {
            Field y = new Field();
            Link link = new Links.ComplexArithmetric.ComplexSquare(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>複素2乗</summary>
    public class ComplexSquare : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexSquare(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexSquare(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexSquareGrad(Y.Grad, X.Value));
            }
        }
    }
}
