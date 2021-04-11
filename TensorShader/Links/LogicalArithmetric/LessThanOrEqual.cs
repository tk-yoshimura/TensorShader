using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>小なりイコール</summary>
        public static Field LessThanOrEqual(Field x1, Field x2) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.LessThanOrEqual(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>小なりイコール</summary>
    public class LessThanOrEqual : LogicalBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public LessThanOrEqual(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LessThanOrEqual(X1.Value, X2.Value));
        }
    }
}
