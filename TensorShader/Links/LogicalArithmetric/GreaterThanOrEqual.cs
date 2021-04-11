using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>大なりイコール</summary>
        public static Field GreaterThanOrEqual(Field x1, Field x2) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.GreaterThanOrEqual(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>大なりイコール</summary>
    public class GreaterThanOrEqual : LogicalBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public GreaterThanOrEqual(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(GreaterThanOrEqual(X1.Value, X2.Value));
        }
    }
}
