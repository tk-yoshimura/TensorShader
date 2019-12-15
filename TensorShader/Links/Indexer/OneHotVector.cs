using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>インデクスをOneHot特徴量に変換</summary>
        public static Field OneHotVector(Field x, int channels) {
            Field y = new Field();
            Link link = new Links.Indexer.OneHotVector(x, y, channels);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Indexer {
    /// <summary>インデクスをOneHot特徴量に変換</summary>
    public class OneHotVector : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>チャネル数</summary>
        protected int Channels { private set; get; }

        /// <summary>コンストラクタ</summary>
        public OneHotVector(Field infield, Field outfield, int channels)
            : base(new Field[] { infield }, outfield) {
            this.Channels = channels;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(OneHotVector(X.Value, Channels));
        }
    }
}
