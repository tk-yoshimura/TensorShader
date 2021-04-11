using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>トリミング</summary>
        public static Field Trimming3D(Field x, int trim) {
            Field y = new();
            Link link = new Links.Connection3D.Trimming(x, y, trim);

            link.Forward();

            return y;
        }

        /// <summary>トリミング</summary>
        public static Field Trimming3D(Field x, int trim_left, int trim_right, int trim_top, int trim_bottom, int trim_front, int trim_rear) {
            Field y = new();
            Link link = new Links.Connection3D.Trimming(x, y, trim_left, trim_right, trim_top, trim_bottom, trim_front, trim_rear);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>トリミング</summary>
    public class Trimming : Link {
        /// <summary>トリミング左幅</summary>
        public int TrimLeft { private set; get; }

        /// <summary>トリミング右幅</summary>
        public int TrimRight { private set; get; }

        /// <summary>トリミング上幅</summary>
        public int TrimTop { private set; get; }

        /// <summary>トリミング下幅</summary>
        public int TrimBottom { private set; get; }

        /// <summary>トリミング前幅</summary>
        public int TrimFront { private set; get; }

        /// <summary>トリミング後幅</summary>
        public int TrimRear { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Trimming(Field infield, Field outfield, int trim)
            : this(infield, outfield, trim, trim, trim, trim, trim, trim) { }

        /// <summary>コンストラクタ</summary>
        public Trimming(Field infield, Field outfield, int trim_left, int trim_right, int trim_top, int trim_bottom, int trim_front, int trim_rear)
            : base(new Field[] { infield }, outfield) {
            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;
            this.TrimTop = trim_top;
            this.TrimBottom = trim_bottom;
            this.TrimFront = trim_front;
            this.TrimRear = trim_rear;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Trimming3D(X.Value, TrimLeft, TrimRight, TrimTop, TrimBottom, TrimFront, TrimRear));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ZeroPadding3D(Y.Grad, TrimLeft, TrimRight, TrimTop, TrimBottom, TrimFront, TrimRear));
            }
        }
    }
}
