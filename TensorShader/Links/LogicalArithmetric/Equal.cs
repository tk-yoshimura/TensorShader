using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>イコール</summary>
        public static Field Equal(Field x1, Field x2) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.Equal(x1, x2, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>イコール</summary>
    public class Equal : LogicalBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Equal(Field infield1, Field infield2, Field outfield)
            : base(infield1, infield2, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Equal(X1.Value, X2.Value));
        }
    }
}
