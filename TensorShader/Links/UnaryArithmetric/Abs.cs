using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>絶対値</summary>
        public static Field Abs(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Abs(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>絶対値</summary>
    internal class Abs : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Abs(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Abs(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Sign(X.Value) * Y.Grad);
            }
        }
    }
}
