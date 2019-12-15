using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>単位ステップ関数</summary>
        public static Field Step(Field x) {
            Field y = new Field();
            Link link = new Links.UnaryArithmetric.Step(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>単位ステップ関数</summary>
    internal class Step : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Step(Field infield, Field outfield)
            : base(infield, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Step(X.Value));
        }
    }
}
