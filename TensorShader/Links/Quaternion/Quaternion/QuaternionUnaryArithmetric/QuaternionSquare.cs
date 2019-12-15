using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>四元数2乗</summary>
        public static Field QuaternionSquare(Field x) {
            Field y = new Field();
            Link link = new Links.QuaternionArithmetric.QuaternionSquare(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionArithmetric {
    /// <summary>四元数2乗</summary>
    public class QuaternionSquare : UnaryArithmetric.UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionSquare(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionSquare(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(QuaternionMulGrad(X.Value, Y.Grad) + QuaternionMulTransposeGrad(X.Value, Y.Grad));
            }
        }
    }
}
