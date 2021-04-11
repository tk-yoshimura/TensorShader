using System.Linq;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>バッチ方向に乗算</summary>
        public static Field BatchwiseMul(Field x, Field v) {
            Field y = new();
            Link link = new Links.ArrayManipulation.BatchwiseMul(x, v, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>バッチ方向に乗算</summary>
    public class BatchwiseMul : Link {
        /// <summary>入力項1</summary>
        protected Field X => InFields[0];

        /// <summary>入力項2</summary>
        protected Field V => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public BatchwiseMul(Field infield, Field vecfield, Field outfield)
            : base(new Field[] { infield, vecfield }, outfield) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(BatchwiseMul(X.Value, V.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(BatchwiseMul(Y.Grad, V.Value));
            }

            if (V.EnableBackprop) {
                int[] axes = (new int[X.Value.Shape.Ndim - 1]).Select((_, idx) => idx).ToArray();

                V.AddGrad(Sum(Y.Grad * X.Value, axes));
            }
        }
    }
}
