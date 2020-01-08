namespace TensorShader {
    public partial class Field {
        /// <summary>分散</summary>
        public static Field Variance(Field x, int[] axes = null, bool keepdims = false) {
            return SquareAverage(x, axes, keepdims) - Square(Average(x, axes, keepdims));
        }

        /// <summary>分散</summary>
        public static Field Variance(Field x, int axis, bool keepdims = false) {
            return Variance(x, new int[]{ axis }, keepdims);
        }
    }
}
