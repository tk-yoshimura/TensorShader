using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>四元数積</summary>
        public static Field QuaternionMul(Field x1, Field x2) {
            Field y = new();
            Link link = new Links.QuaternionArithmetric.QuaternionMul(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>四元数積</summary>
    public class QuaternionMul : BinaryArithmetric.BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionMul(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionMul(X1.Value, X2.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(QuaternionMulGrad(X2.Value, Y.Grad));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(QuaternionMulTransposeGrad(X1.Value, Y.Grad));
            }
        }
    }
}
