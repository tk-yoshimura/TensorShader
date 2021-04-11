using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>正接関数</summary>
        public static Field Tan(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Tan(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>正接関数</summary>
    internal class Tan : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Tan(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Tan(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Rcp(Square(Cos(X.Value))) * Y.Grad);
            }
        }
    }
}
