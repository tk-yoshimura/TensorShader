using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>正答率</summary>
        public static Field Accuracy(Field x, Field t) {
            Field y = new();
            Link link = new Links.Evaluation.Classify.Accuracy(x, t, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Evaluation.Classify {
    /// <summary>正答率</summary>
    public class Accuracy : ClassifyEvaluation {
        /// <summary>コンストラクタ</summary>
        public Accuracy(Field xfield, Field tfield, Field yfield)
            : base(xfield, tfield, yfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Sum(OneHotVector(ArgMax(X.Value), X.Shape.Channels) * OneHotVector(T.Value, X.Shape.Channels)) / X.Shape.Batch);
        }
    }
}
