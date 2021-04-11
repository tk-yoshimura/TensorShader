using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>双曲線正弦関数</summary>
        public static Field Sinh(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Sinh(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>双曲線正弦関数</summary>
    internal class Sinh : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sinh(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Sinh(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Cosh(X.Value) * Y.Grad);
            }
        }
    }
}
