using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素積</summary>
        public static Field ComplexMul(Field x1, Field x2) {
            Field y = new Field();
            Link link = new Links.ComplexArithmetric.ComplexMul(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexArithmetric {
    /// <summary>複素積</summary>
    public class ComplexMul : BinaryArithmetric.BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexMul(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexMul(X1.Value, X2.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(ComplexMulGrad(Y.Grad, X2.Value));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(ComplexMulGrad(Y.Grad, X1.Value));
            }
        }
    }
}
