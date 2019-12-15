namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ソート</summary>
        public static VariableNode ArgSort(VariableNode x, int axis) {
            InputNode index = Tensor.Index(x.Shape, axis);

            return SortWithKey(x, index, axis).value;
        }
    }

    public partial class Tensor {
        /// <summary>ソート</summary>
        public static Tensor ArgSort(Tensor x, int axis) {
            Tensor index = Index(x.Shape, axis);

            return SortWithKey(x, index, axis).value;
        }
    }
}
