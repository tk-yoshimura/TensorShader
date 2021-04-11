using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>逆正弦関数</summary>
        public static Field Arcsin(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Arcsin(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>逆正弦関数</summary>
    internal class Arcsin : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Arcsin(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Arcsin(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Rsqrt(1 - Square(X.Value)) * Y.Grad);
            }
        }
    }
}
