using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>2次元逆畳み込み</summary>
        public static Field Deconvolution2D(Field x, Field w, int stride, Shape outshape = null) {
            Field y = new Field();
            Link link = new Links.Connection2D.Deconvolution(x, w, y, stride, outshape);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection2D {
    /// <summary>2次元逆畳み込み</summary>
    public class Deconvolution : Link {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>出力形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Deconvolution(Field infield, Field kernelfield, Field outfield, int stride, Shape outshape)
            : base(new Field[] { infield, kernelfield }, outfield) {
            this.Stride = stride;
            this.OutShape = outshape;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Deconvolution2D(X.Value, W.Value, Stride, OutShape));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Convolution2D(Y.Grad, W.Value, Stride));
            }

            if (W.EnableBackprop) {
                W.AddGrad(KernelProduct2D(Y.Grad, X.Value, W.Shape.Width, W.Shape.Height, Stride));
            }
        }
    }
}
