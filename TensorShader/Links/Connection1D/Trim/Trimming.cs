using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>トリミング</summary>
        public static Field Trimming1D(Field x, int trim) {
            Field y = new Field();
            Link link = new Links.Connection1D.Trimming(x, y, trim);

            link.Forward();

            return y;
        }

        /// <summary>トリミング</summary>
        public static Field Trimming1D(Field x, int trim_left, int trim_right) {
            Field y = new Field();
            Link link = new Links.Connection1D.Trimming(x, y, trim_left, trim_right);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>トリミング</summary>
    public class Trimming : Link {
        /// <summary>トリミング左幅</summary>
        public int TrimLeft { private set; get; }

        /// <summary>トリミング右幅</summary>
        public int TrimRight { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Trimming(Field infield, Field outfield, int trim)
            : this(infield, outfield, trim, trim) { }

        /// <summary>コンストラクタ</summary>
        public Trimming(Field infield, Field outfield, int trim_left, int trim_right)
            : base(new Field[] { infield }, outfield) {
            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Trimming1D(X.Value, TrimLeft, TrimRight));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ZeroPadding1D(Y.Grad, TrimLeft, TrimRight));
            }
        }
    }
}
