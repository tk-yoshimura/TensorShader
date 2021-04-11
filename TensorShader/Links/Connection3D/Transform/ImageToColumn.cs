using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ImageToColumn変換</summary>
        public static Field ImageToColumn3D(Field x, int kwidth, int kheight, int kdepth) {
            Field y = new();
            Link link = new Links.Connection3D.ImageToColumn(x, y, kwidth, kheight, kdepth);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>ImageToColumn変換</summary>
    public class ImageToColumn : Link {

        /// <summary>フィルタサイズ</summary>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public int KernelHeight { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public int KernelDepth { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ImageToColumn(Field infield, Field outfield, int kwidth, int kheight, int kdepth)
            : base(new Field[] { infield }, outfield) {

            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ImageToColumn3D(X.Value, KernelWidth, KernelHeight, KernelDepth));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ColumnToImage3D(Y.Grad, KernelWidth, KernelHeight, KernelDepth));
            }
        }
    }
}
