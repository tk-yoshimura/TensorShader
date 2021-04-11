using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>否定</summary>
        public static Field Not(Field x) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.Not(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>否定</summary>
    public class Not : LogicalUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Not(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LogicalNot(X.Value));
        }
    }
}
