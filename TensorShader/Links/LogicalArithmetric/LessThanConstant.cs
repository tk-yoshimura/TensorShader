using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>小なり</summary>
        public static Field LessThan(Field x, float c) {
            Field y = new Field();
            Link link = new Links.LogicalArithmetric.LessThanConstant(x, y, c);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>小なり</summary>
    public class LessThanConstant : LogicalUnaryArithmetric {
        /// <summary>定数項</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public LessThanConstant(Field infield, Field outfield, float c)
            : base(infield, outfield) {
            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LessThan(X.Value, Constant));
        }
    }
}
