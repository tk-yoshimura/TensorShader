using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>クラスごとの正答数</summary>
        public static Field Matches(Field x, Field t) {
            Field y = new();
            Link link = new Links.Evaluation.Classify.Matches(x, t, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Evaluation.Classify {
    /// <summary>クラスごとの正答数</summary>
    public class Matches : ClassifyEvaluation {
        /// <summary>コンストラクタ</summary>
        public Matches(Field xfield, Field tfield, Field yfield)
            : base(xfield, tfield, yfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            int channels = X.Shape.Channels;

            VariableNode x_onehot = OneHotVector(ArgMax(X.Value), channels);
            VariableNode t_onehot = OneHotVector(T.Value, channels);

            Y.AssignValue(Sum(x_onehot * t_onehot, axes: new int[] { Axis.Map0D.Batch }));
        }
    }
}
