using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>非数をゼロとして返す</summary>
        public static Field NanAsZero(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.NanAsZero(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>非数をゼロとして返す</summary>
    internal class NanAsZero : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public NanAsZero(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(NanAsZero(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad((1 - IsNan(X.Value)) * Y.Grad);
            }
        }
    }
}
