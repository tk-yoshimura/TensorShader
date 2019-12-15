using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>混同行列</summary>
        public static Field ConfusionMatrix(Field x, Field t) {
            Field y = new Field();
            Link link = new Links.Evaluation.Classify.ConfusionMatrix(x, t, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Evaluation.Classify {
    /// <summary>混同行列</summary>
    public class ConfusionMatrix : ClassifyEvaluation {
        /// <summary>コンストラクタ</summary>
        public ConfusionMatrix(Field xfield, Field tfield, Field yfield)
            : base(xfield, tfield, yfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            int channels = X.Shape.Channels, batch = X.Shape.Batch;

            VariableNode x_onehot = OneHotVector(ArgMax(X.Value), channels);
            VariableNode t_onehot = OneHotVector(T.Value, channels);

            VariableNode x_table =
                Broadcast(
                    Reshape(x_onehot, new Shape(ShapeType.Matrix, channels, 1, batch)),
                    new Shape(ShapeType.Matrix, channels, channels, batch)
                );

            VariableNode t_table =
                Broadcast(
                    Reshape(t_onehot, new Shape(ShapeType.Matrix, 1, channels, batch)),
                    new Shape(ShapeType.Matrix, channels, channels, batch)
                );

            Y.AssignValue(Sum(x_table * t_table, axes: new int[] { 2 }));
        }
    }
}
