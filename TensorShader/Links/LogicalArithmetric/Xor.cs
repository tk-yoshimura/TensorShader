using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>Xor</summary>
        public static Field Xor(Field x1, Field x2) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.Xor(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>Xor</summary>
    public class Xor : LogicalBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Xor(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LogicalXor(X1.Value, X2.Value));
        }
    }
}
