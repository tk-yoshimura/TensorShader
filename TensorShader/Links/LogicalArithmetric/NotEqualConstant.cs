using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ノットイコール</summary>
        public static Field NotEqual(Field x, float c) {
            Field y = new();
            Link link = new Links.LogicalArithmetric.NotEqualConstant(x, y, c);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>ノットイコール</summary>
    public class NotEqualConstant : LogicalUnaryArithmetric {
        /// <summary>定数項</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public NotEqualConstant(Field infield, Field outfield, float c)
            : base(infield, outfield) {

            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(NotEqual(X.Value, Constant));
        }
    }
}
