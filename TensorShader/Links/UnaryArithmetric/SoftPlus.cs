using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>SoftPlus</summary>
        public static Field SoftPlus(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.SoftPlus(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>SoftPlus</summary>
    internal class SoftPlus : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public SoftPlus(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(SoftPlus(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Sigmoid(X.Value) * Y.Grad);
            }
        }
    }
}
