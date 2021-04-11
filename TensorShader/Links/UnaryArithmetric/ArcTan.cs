using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>逆正接関数</summary>
        public static Field Arctan(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Arctan(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>逆正接関数</summary>
    internal class Arctan : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Arctan(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Arctan(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Rcp(1 + Square(X.Value)) * Y.Grad);
            }
        }
    }
}
