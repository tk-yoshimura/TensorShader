using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>1次元ゼロパディング</summary>
        public static Field ZeroPadding1D(Field x, int pad) {
            Field y = new Field();
            Link link = new Links.Connection1D.ZeroPadding(x, y, pad);

            link.Forward();

            return y;
        }

        /// <summary>1次元ゼロパディング</summary>
        public static Field ZeroPadding1D(Field x, int pad_left, int pad_right) {
            Field y = new Field();
            Link link = new Links.Connection1D.ZeroPadding(x, y, pad_left, pad_right);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>1次元ゼロパディング</summary>
    public class ZeroPadding : Link {
        /// <summary>パディング左幅</summary>
        public int PadLeft { private set; get; }

        /// <summary>パディング右幅</summary>
        public int PadRight { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ZeroPadding(Field infield, Field outfield, int pad)
            : this(infield, outfield, pad, pad) { }

        /// <summary>コンストラクタ</summary>
        public ZeroPadding(Field infield, Field outfield, int pad_left, int pad_right)
            : base(new Field[] { infield }, outfield) {
            this.PadLeft = pad_left;
            this.PadRight = pad_right;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ZeroPadding1D(X.Value, PadLeft, PadRight));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Trimming1D(Y.Grad, PadLeft, PadRight));
            }
        }
    }
}
