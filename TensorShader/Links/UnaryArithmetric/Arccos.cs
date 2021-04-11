using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>逆余弦関数</summary>
        public static Field Arccos(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Arccos(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>逆余弦関数</summary>
    internal class Arccos : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Arccos(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Arccos(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(-Rsqrt(1 - Square(X.Value)) * Y.Grad);
            }
        }
    }
}
