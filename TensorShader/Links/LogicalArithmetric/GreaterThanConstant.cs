using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>大なり</summary>
        public static Field GreaterThan(Field x, float c) {
            Field y = new Field();
            Link link = new Links.LogicalArithmetric.GreaterThanConstant(x, y, c);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>大なり</summary>
    public class GreaterThanConstant : LogicalUnaryArithmetric {
        /// <summary>定数項</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public GreaterThanConstant(Field infield, Field outfield, float c)
            : base(infield, outfield) {
            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(GreaterThan(X.Value, Constant));
        }
    }
}
