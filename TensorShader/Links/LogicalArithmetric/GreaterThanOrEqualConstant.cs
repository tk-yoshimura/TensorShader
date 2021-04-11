using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>大なりイコール</summary>
        public static Field GreaterThanOrEqual(Field x, float c) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.GreaterThanOrEqualConstant(x, y, c);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>大なりイコール</summary>
    public class GreaterThanOrEqualConstant : LogicalUnaryArithmetric {
        /// <summary>定数項</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public GreaterThanOrEqualConstant(Field infield, Field outfield, float c)
            : base(infield, outfield) {

            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(GreaterThanOrEqual(X.Value, Constant));
        }
    }
}
