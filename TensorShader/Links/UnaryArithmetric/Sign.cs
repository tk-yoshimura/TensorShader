using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>符号</summary>
        public static Field Sign(Field x) {
            Field y = new();
            Link link = new Links.UnaryArithmetric.Sign(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>符号</summary>
    internal class Sign : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Sign(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Sign(X.Value));
        }
    }
}
