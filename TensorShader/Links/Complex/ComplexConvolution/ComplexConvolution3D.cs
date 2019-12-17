using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素3次元畳み込み</summary>
        public static Field ComplexConvolution3D(Field x, Field w) {
            Field y = new Field();
            Link link = new Links.ComplexConvolution.ComplexConvolution3D(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexConvolution {
    /// <summary>複素3次元畳み込み</summary>
    public class ComplexConvolution3D : Link {

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ComplexConvolution3D(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexConvolution3D(X.Value, W.Value, gradmode: false));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexDeconvolution3D(Y.Grad, W.Value, gradmode: true));
            }

            if (W.EnableBackprop) {
                W.AddGrad(ComplexKernelProduct3D(X.Value, Y.Grad, W.Shape.Width, W.Shape.Height, W.Shape.Depth, transpose: false));
            }
        }
    }
}
