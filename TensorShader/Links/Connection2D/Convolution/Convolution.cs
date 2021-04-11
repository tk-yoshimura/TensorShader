using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>2次元畳み込み</summary>
        public static Field Convolution2D(Field x, Field w) {
            Field y = new();
            Link link = new Links.Connection2D.Convolution(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection2D {
    /// <summary>2次元畳み込み</summary>
    public class Convolution : Link {

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public Convolution(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Convolution2D(X.Value, W.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Deconvolution2D(Y.Grad, W.Value));
            }

            if (W.EnableBackprop) {
                W.AddGrad(KernelProduct2D(X.Value, Y.Grad, W.Shape.Width, W.Shape.Height));
            }
        }
    }
}
