using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ベクトル四元数回転積</summary>
        public static Field TrivectorQuaternionMul(Field v, Field q) {
            Field y = new();
            Link link = new Links.QuaternionArithmetric.TrivectorQuaternionMul(v, q, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>3次元ベクトル四元数回転積</summary>
    public class TrivectorQuaternionMul : BinaryArithmetric.BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorQuaternionMul(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorQuaternionMul(X1.Value, X2.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X1.EnableBackprop) {
                X1.AddGrad(TrivectorQuaternionMulVGrad(Y.Grad, X2.Value));
            }

            if (X2.EnableBackprop) {
                X2.AddGrad(TrivectorQuaternionMulQGrad(X1.Value, Y.Grad, X2.Value));
            }
        }
    }
}
