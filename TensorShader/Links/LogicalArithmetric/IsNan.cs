using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>非数判定</summary>
        public static Field IsNan(Field x) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.IsNan(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>非数判定</summary>
    public class IsNan : LogicalUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public IsNan(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(IsNan(X.Value));
        }
    }
}
