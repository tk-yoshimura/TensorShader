using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>小なり</summary>
        public static Field LessThan(Field x1, Field x2) {
            Field y = new Field();
            Link link = new Links.LogicalArithmetric.LessThan(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>小なり</summary>
    public class LessThan : LogicalBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public LessThan(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LessThan(X1.Value, X2.Value));
        }
    }
}
