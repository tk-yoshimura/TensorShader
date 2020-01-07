using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ColumnToImage変換</summary>
        public static Field ColumnToImage3D(Field x, int kwidth, int kheight, int kdepth) {
            Field y = new Field();
            Link link = new Links.Connection3D.ColumnToImage(x, y, kwidth, kheight, kdepth);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>ColumnToImage変換</summary>
    public class ColumnToImage : Link {

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
        public ColumnToImage(Field infield, Field outfield, int kwidth, int kheight, int kdepth)
            : base(new Field[] { infield }, outfield) {

            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ColumnToImage3D(X.Value, KernelWidth, KernelHeight, KernelDepth));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ImageToColumn3D(Y.Grad, KernelWidth, KernelHeight, KernelDepth));
            }
        }
    }
}
