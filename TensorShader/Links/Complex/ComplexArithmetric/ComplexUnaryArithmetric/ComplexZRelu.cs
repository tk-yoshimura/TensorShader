using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ComplexZRelu</summary>
        /// <remarks>ガウス平面第1象限以外を0とする</remarks>
        public static Field ComplexZRelu(Field x) {
            Field y = new();
            Link link = new Links.ComplexArithmetric.ComplexZRelu(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>ComplexZRelu</summary>
    public class ComplexZRelu : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexZRelu(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexZRelu(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexZReluGrad(Y.Grad, X.Value));
            }
        }
    }
}
