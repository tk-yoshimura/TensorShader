using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>論理和</summary>
        /// <remarks>Lotfi A.Zadehのファジー論理和に相当</remarks>
        public static Field Or(Field x1, Field x2) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.Or(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>論理和</summary>
    /// <remarks>Lotfi A.Zadehのファジー論理和に相当</remarks>
    public class Or : LogicalBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Or(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LogicalOr(X1.Value, X2.Value));
        }
    }
}
