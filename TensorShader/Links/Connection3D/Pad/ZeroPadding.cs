using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ゼロパディング</summary>
        public static Field ZeroPadding3D(Field x, int pad) {
            Field y = new Field();
            Link link = new Links.Connection3D.ZeroPadding(x, y, pad);

            link.Forward();

            return y;
        }

        /// <summary>3次元ゼロパディング</summary>
        public static Field ZeroPadding3D(Field x, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_rear) {
            Field y = new Field();
            Link link = new Links.Connection3D.ZeroPadding(x, y, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>3次元ゼロパディング</summary>
    public class ZeroPadding : Link {
        /// <summary>パディング左幅</summary>
        public int PadLeft { private set; get; }

        /// <summary>パディング右幅</summary>
        public int PadRight { private set; get; }

        /// <summary>パディング上幅</summary>
        public int PadTop { private set; get; }

        /// <summary>パディング下幅</summary>
        public int PadBottom { private set; get; }

        /// <summary>パディング前幅</summary>
        public int PadFront { private set; get; }

        /// <summary>パディング後幅</summary>
        public int PadRear { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ZeroPadding(Field infield, Field outfield, int pad)
            : this(infield, outfield, pad, pad, pad, pad, pad, pad) { }

        /// <summary>コンストラクタ</summary>
        public ZeroPadding(Field infield, Field outfield, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_rear)
            : base(new Field[] { infield }, outfield) {
            this.PadLeft = pad_left;
            this.PadRight = pad_right;
            this.PadTop = pad_top;
            this.PadBottom = pad_bottom;
            this.PadFront = pad_front;
            this.PadRear = pad_rear;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ZeroPadding3D(X.Value, PadLeft, PadRight, PadTop, PadBottom, PadFront, PadRear));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Trimming3D(Y.Grad, PadLeft, PadRight, PadTop, PadBottom, PadFront, PadRear));
            }
        }
    }
}
