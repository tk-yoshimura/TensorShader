using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>空間次元をチャネル次元に展開</summary>
        public static Field SpaceToChannel3D(Field x, int scale) {
            Field y = new Field();
            Link link = new Links.Connection3D.SpaceToChannel(x, y, scale);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>空間次元をチャネル次元に展開</summary>
    public class SpaceToChannel : Link {
        /// <summary>倍率</summary>
        public int Scale { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public SpaceToChannel(Field infield, Field outfield, int scale)
            : base(new Field[] { infield }, outfield) {
            this.Scale = scale;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(SpaceToChannel3D(X.Value, Scale));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ChannelToSpace3D(Y.Grad, Scale));
            }
        }
    }
}
