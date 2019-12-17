using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>四元数1次元逆畳み込み</summary>
        public static Field QuaternionDeconvolution1D(Field x, Field w) {
            Field y = new Field();
            Link link = new Links.QuaternionConvolution.QuaternionDeconvolution1D(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.QuaternionConvolution {
    /// <summary>四元数1次元逆畳み込み</summary>
    public class QuaternionDeconvolution1D : Link {

        /// <summary>出力形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public QuaternionDeconvolution1D(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(QuaternionDeconvolution1D(X.Value, W.Value, gradmode: false));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(QuaternionConvolution1D(Y.Grad, W.Value, gradmode: true));
            }

            if (W.EnableBackprop) {
                W.AddGrad(QuaternionKernelProduct1D(Y.Grad, X.Value, W.Shape.Width, transpose: true));
            }
        }
    }
}
