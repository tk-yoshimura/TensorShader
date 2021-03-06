using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>チャネル次元を空間方向に展開</summary>
        public static Field ChannelToSpace1D(Field x, int scale) {
            Field y = new();
            Link link = new Links.Connection1D.ChannelToSpace(x, y, scale);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>チャネル次元を空間方向に展開</summary>
    public class ChannelToSpace : Link {
        /// <summary>倍率</summary>
        public int Scale { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ChannelToSpace(Field infield, Field outfield, int scale)
            : base(new Field[] { infield }, outfield) {
            this.Scale = scale;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ChannelToSpace1D(X.Value, Scale));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(SpaceToChannel1D(Y.Grad, Scale));
            }
        }
    }
}
