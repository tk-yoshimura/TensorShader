using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素虚部</summary>
        public static Field ComplexImag(Field x) {
            Field y = new();
            Link link = new Links.ComplexArithmetric.ComplexImag(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>複素虚部</summary>
    public class ComplexImag : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexImag(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexImag(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexPureImag(Y.Grad));
            }
        }
    }
}
