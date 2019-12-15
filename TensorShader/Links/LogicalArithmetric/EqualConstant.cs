using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>イコール</summary>
        public static Field Equal(Field x, float c) {
            Field y = new Field();
            Link link = new Links.LogicalArithmetric.EqualConstant(x, y, c);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>イコール</summary>
    public class EqualConstant : LogicalUnaryArithmetric {
        /// <summary>定数項</summary>
        public float Constant { private set; get; }

        /// <summary>コンストラクタ</summary>
        public EqualConstant(Field infield, Field outfield, float c)
            : base(infield, outfield) {
            this.Constant = c;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Equal(X.Value, Constant));
        }
    }
}
