using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>切り捨て関数</summary>
        public static Field Floor(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Floor(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>切り捨て関数</summary>
    internal class Floor : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Floor(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Floor(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad);
            }
        }
    }
}
