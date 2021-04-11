using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>正弦関数</summary>
        public static Field Sin(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Sin(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>正弦関数</summary>
    internal class Sin : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sin(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Sin(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Cos(X.Value) * Y.Grad);
            }
        }
    }
}
