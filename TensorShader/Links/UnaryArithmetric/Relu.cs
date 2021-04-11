using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>Relu</summary>
        public static Field Relu(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Relu(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>Relu</summary>
    internal class Relu : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Relu(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Relu(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ReluGrad(Y.Grad, X.Value));
            }
        }
    }
}
