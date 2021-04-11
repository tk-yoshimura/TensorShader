using System.Linq;
using static TensorShader.VariableNode;

namespace TensorShader.Links.Aggregation {
    /// <summary>統計</summary>
    public abstract class Aggregation : Link {
        /// <summary>軸</summary>
        protected int[] Axes { private set; get; }

        /// <summary>集約軸を保持するか</summary>
        protected bool KeepDims { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        private readonly Shape keepdims_shape;

        /// <summary>コンストラクタ</summary>
        public Aggregation(Field infield, Field outfield, int[] axes = null, bool keepdims = false)
            : base(new Field[] { infield }, outfield) {
            if (axes is null) {
                axes = (new int[infield.Shape.Ndim]).Select((_, idx) => idx).ToArray();
            }

            this.Axes = axes;
            this.KeepDims = keepdims;

            int[] s = infield.Shape;

            foreach (int axis in axes) {
                s[axis] = 1;
            }

            keepdims_shape = new Shape(infield.Shape.Type, s);
        }

        /// <summary>形状を調整する</summary>
        protected VariableNode AdjustShape(VariableNode node, Shape shape) {
            if (!KeepDims) {
                node = Reshape(node, keepdims_shape);
            }

            return Broadcast(node, shape);
        }
    }
}
