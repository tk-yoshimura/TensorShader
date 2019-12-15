using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>次元ごとに最大インデクスを抽出</summary>
        public static Field ArgMax(Field x) {
            Field y = new Field();
            Link link = new Links.Indexer.ArgMax(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Indexer {
    /// <summary>次元ごとに最大インデクスを抽出</summary>
    public class ArgMax : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ArgMax(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ArgMax(X.Value));
        }
    }
}
