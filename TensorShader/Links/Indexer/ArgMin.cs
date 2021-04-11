using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>次元ごとに最小インデクスを抽出</summary>
        public static Field ArgMin(Field x) {
            Field y = new();
            Link link = new Links.Indexer.ArgMin(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Indexer {
    /// <summary>次元ごとに最小インデクスを抽出</summary>
    public class ArgMin : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ArgMin(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ArgMin(X.Value));
        }
    }
}
