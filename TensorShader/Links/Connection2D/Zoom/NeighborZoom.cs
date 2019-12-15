using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>最近傍補間</summary>
        /// <remarks>倍率2固定</remarks>
        public static Field NeighborZoom2D(Field x) {
            Field y = new Field();
            Link link = new Links.Connection2D.NeighborZoom(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection2D {
    /// <summary>最近傍補間</summary>
    /// <remarks>倍率2固定</remarks>
    public class NeighborZoom : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public NeighborZoom(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) {
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(NeighborZoom2D(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(AveragePooling2D(Y.Grad, stride: 2));
            }
        }
    }
}
