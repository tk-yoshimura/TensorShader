using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>2次元エッジパディング</summary>
        public static Field EdgePadding2D(Field x, int pad) {
            Field y = new Field();
            Link link = new Links.Connection2D.EdgePadding(x, y, pad);

            link.Forward();

            return y;
        }

        /// <summary>2次元エッジパディング</summary>
        public static Field EdgePadding2D(Field x, int pad_left, int pad_right, int pad_top, int pad_bottom) {
            Field y = new Field();
            Link link = new Links.Connection2D.EdgePadding(x, y, pad_left, pad_right, pad_top, pad_bottom);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection2D {
    /// <summary>2次元エッジパディング</summary>
    public class EdgePadding : Link {
        /// <summary>パディング左幅</summary>
        public int PadLeft { private set; get; }

        /// <summary>パディング右幅</summary>
        public int PadRight { private set; get; }

        /// <summary>パディング上幅</summary>
        public int PadTop { private set; get; }

        /// <summary>パディング下幅</summary>
        public int PadBottom { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public EdgePadding(Field infield, Field outfield, int pad)
            : this(infield, outfield, pad, pad, pad, pad) { }

        /// <summary>コンストラクタ</summary>
        public EdgePadding(Field infield, Field outfield, int pad_left, int pad_right, int pad_top, int pad_bottom)
            : base(new Field[] { infield }, outfield) {
            this.PadLeft = pad_left;
            this.PadRight = pad_right;
            this.PadTop = pad_top;
            this.PadBottom = pad_bottom;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(EdgePadding2D(X.Value, PadLeft, PadRight, PadTop, PadBottom));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Trimming2D(Y.Grad, PadLeft, PadRight, PadTop, PadBottom));
            }
        }
    }
}
