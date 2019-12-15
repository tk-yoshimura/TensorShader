using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ノットイコール</summary>
        public static Field NotEqual(Field x1, Field x2) {
            Field y = new Field();
            Link link = new Links.LogicalArithmetric.NotEqual(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>ノットイコール</summary>
    public class NotEqual : LogicalBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public NotEqual(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(NotEqual(X1.Value, X2.Value));
        }
    }
}
