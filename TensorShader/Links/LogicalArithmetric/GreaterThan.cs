using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>大なり</summary>
        public static Field GreaterThan(Field x1, Field x2) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.GreaterThan(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>大なり</summary>
    public class GreaterThan : LogicalBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public GreaterThan(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(GreaterThan(X1.Value, X2.Value));
        }
    }
}
