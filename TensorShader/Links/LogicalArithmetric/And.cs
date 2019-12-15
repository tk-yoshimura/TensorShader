using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>論理積</summary>
        /// <remarks>Lotfi A.Zadehのファジー論理積に相当</remarks>
        public static Field And(Field x1, Field x2) {
            Field y = new Field();
            Link link = new Links.LogicalArithmetric.And(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>論理積</summary>
    /// <remarks>Lotfi A.Zadehのファジー論理積に相当</remarks>
    public class And : LogicalBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public And(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LogicalAnd(X1.Value, X2.Value));
        }
    }
}
