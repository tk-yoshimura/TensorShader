using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ColumnToImage変換</summary>
        public static Field ColumnToImage1D(Field x, int kwidth) {
            Field y = new Field();
            Link link = new Links.Connection1D.ColumnToImage(x, y, kwidth);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>ColumnToImage変換</summary>
    public class ColumnToImage : Link {

        /// <summary>フィルタサイズ</summary>
        public int KernelWidth { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ColumnToImage(Field infield, Field outfield, int kwidth)
            : base(new Field[] { infield }, outfield) {

            this.KernelWidth = kwidth;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ColumnToImage1D(X.Value, KernelWidth));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ImageToColumn1D(Y.Grad, KernelWidth));
            }
        }
    }
}
