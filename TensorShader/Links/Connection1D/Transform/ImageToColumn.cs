using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ImageToColumn変換</summary>
        public static Field ImageToColumn1D(Field x, int kwidth) {
            Field y = new();
            Link link = new Links.Connection1D.ImageToColumn(x, y, kwidth);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>ImageToColumn変換</summary>
    public class ImageToColumn : Link {

        /// <summary>フィルタサイズ</summary>
        public int KernelWidth { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ImageToColumn(Field infield, Field outfield, int kwidth)
            : base(new Field[] { infield }, outfield) {

            this.KernelWidth = kwidth;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ImageToColumn1D(X.Value, KernelWidth));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ColumnToImage1D(Y.Grad, KernelWidth));
            }
        }
    }
}
