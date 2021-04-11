using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>シグモイド関数</summary>
        public static Field Sigmoid(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Sigmoid(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>シグモイド関数</summary>
    internal class Sigmoid : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sigmoid(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Sigmoid(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Value * (1 - Y.Value) * Y.Grad);
            }
        }
    }
}
