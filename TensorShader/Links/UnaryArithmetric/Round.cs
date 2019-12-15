using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>最近傍整数丸め関数</summary>
        public static Field Round(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Round(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>最近傍整数丸め関数</summary>
    internal class Round : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Round(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Round(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad);
            }
        }
    }
}
