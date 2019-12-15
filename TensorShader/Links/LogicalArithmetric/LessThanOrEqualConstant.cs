using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>小なりイコール</summary>
        public static Field LessThanOrEqual(Field x, float c) {
            Field y = new Field();
            Link link = new Links.LogicalArithmetric.LessThanOrEqualConstant(x, y, c);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>小なりイコール</summary>
    public class LessThanOrEqualConstant : LogicalUnaryArithmetric {
        /// <summary>定数項</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public LessThanOrEqualConstant(Field infield, Field outfield, float c)
            : base(infield, outfield) {
            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(LessThanOrEqual(X.Value, Constant));
        }
    }
}
