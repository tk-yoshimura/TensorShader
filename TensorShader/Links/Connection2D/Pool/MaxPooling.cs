using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>最大値プーリング</summary>
        public static Field MaxPooling2D(Field x, int stride) {
            Field y = new Field();
            Link link = new Links.Connection2D.MaxPooling(x, y, stride);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection2D {
    /// <summary>最大値プーリング</summary>
    public class MaxPooling : Link {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public MaxPooling(Field infield, Field outfield, int stride)
            : base(new Field[] { infield }, outfield) {
            this.Stride = stride;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(MaxPooling2D(X.Value, Stride));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(MaxUnpooling2D(Y.Grad, X.Value, Y.Value, Stride));
            }
        }
    }
}
