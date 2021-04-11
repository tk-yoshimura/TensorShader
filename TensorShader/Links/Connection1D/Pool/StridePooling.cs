using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>1次元ストライドプーリング</summary>
        public static Field StridePooling1D(Field x, int stride) {
            Field y = new();
            Link link = new Links.Connection1D.StridePooling(x, y, stride);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>1次元ストライドプーリング</summary>
    public class StridePooling : Link {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public StridePooling(Field infield, Field outfield, int stride)
            : base(new Field[] { infield }, outfield) {
            this.Stride = stride;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(StridePooling1D(X.Value, Stride));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(StrideUnpooling1D(Y.Grad, Stride, X.Shape));
            }
        }
    }
}
