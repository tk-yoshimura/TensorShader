using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ComplexRRelu</summary>
        /// <remarks>実部のみReluを適用</remarks>
        public static Field ComplexRRelu(Field x) {
            Field y = new Field();
            Link link = new Links.ComplexArithmetric.ComplexRRelu(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>ComplexRRelu</summary>
    public class ComplexRRelu : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexRRelu(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexRRelu(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexRReluGrad(Y.Grad, X.Value));
            }
        }
    }
}
