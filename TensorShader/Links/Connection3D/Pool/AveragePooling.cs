using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元平均値プーリング</summary>
        public static Field AveragePooling3D(Field x, int stride) {
            Field y = new();
            Link link = new Links.Connection3D.AveragePooling(x, y, stride);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>3次元平均値プーリング</summary>
    public class AveragePooling : Link {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public AveragePooling(Field infield, Field outfield, int stride)
            : base(new Field[] { infield }, outfield) {
            this.Stride = stride;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(AveragePooling3D(X.Value, Stride));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(AverageUnpooling3D(Y.Grad, Stride, X.Shape));
            }
        }
    }
}
