using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ベクトル3次元畳み込み</summary>
        public static Field TrivectorConvolution3D(Field x, Field w) {
            Field y = new Field();
            Link link = new Links.TrivectorConvolution.TrivectorConvolution3D(x, w, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.TrivectorConvolution {
    /// <summary>3次元ベクトル3次元畳み込み</summary>
    public class TrivectorConvolution3D : Link {

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public TrivectorConvolution3D(Field infield, Field kernelfield, Field outfield)
            : base(new Field[] { infield, kernelfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorConvolution3D(X.Value, W.Value, gradmode: false));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(TrivectorDeconvolution3D(Y.Grad, W.Value, gradmode: true));
            }

            if (W.EnableBackprop) {
                W.AddGrad(TrivectorKernelProduct3D(X.Value, Y.Grad, W.Value, W.Shape.Width, W.Shape.Height, W.Shape.Depth, transpose: false));
            }
        }
    }
}
