using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素実部</summary>
        public static Field ComplexReal(Field x) {
            Field y = new();
            Link link = new Links.ComplexArithmetric.ComplexReal(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>複素実部</summary>
    public class ComplexReal : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexReal(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexReal(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexPureReal(Y.Grad));
            }
        }
    }
}
