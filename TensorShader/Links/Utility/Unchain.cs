namespace TensorShader {
    public partial class Field {
        /// <summary>逆伝搬をブロック</summary>
        public static Field Unchain(Field x) {
            Field y = new();
            Link link = new Links.Utility.Unchain(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Utility {
    /// <summary>逆伝搬をブロック</summary>
    public class Unchain : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Unchain(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(X.Value);
        }
    }
}
