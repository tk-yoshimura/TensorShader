using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ImageToColumn変換</summary>
        public static Field ImageToColumn2D(Field x, int kwidth, int kheight) {
            Field y = new Field();
            Link link = new Links.Connection2D.ImageToColumn(x, y, kwidth, kheight);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection2D {
    /// <summary>ImageToColumn変換</summary>
    public class ImageToColumn : Link {

        /// <summary>フィルタサイズ</summary>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public int KernelHeight { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ImageToColumn(Field infield, Field outfield, int kwidth, int kheight)
            : base(new Field[] { infield }, outfield) {

            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ImageToColumn2D(X.Value, KernelWidth, KernelHeight));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ColumnToImage2D(Y.Grad, KernelWidth, KernelHeight));
            }
        }
    }
}
