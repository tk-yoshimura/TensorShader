using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>チャネル方向に切り出し</summary>
        public static Field ExtractChannel(Field x, int index, int channels = 1) {
            Field y = new();
            Link link = new Links.ArrayManipulation.ExtractChannel(index, channels, x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>チャネル方向に切り出し</summary>
    public class ExtractChannel : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>インデクス</summary>
        public int Index { private set; get; }

        /// <summary>切り出しチャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ExtractChannel(int index, int channels, Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) {
            this.Index = index;
            this.Channels = channels;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ExtractChannel(X.Value, Index, Channels));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(InsertChannel(Y.Grad, Index, X.Shape.Channels));
            }
        }
    }
}
