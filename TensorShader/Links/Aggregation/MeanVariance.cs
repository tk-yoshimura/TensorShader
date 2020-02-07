namespace TensorShader {
    public partial class Field {
        /// <summary>平均/分散</summary>
        public static (Field mean, Field variance) AverageVariance(Field x, int[] axes = null, bool keepdims = false) {
            Field mean = Average(x, axes, keepdims);
            Field variance = SquareAverage(x, axes, keepdims) - Square(mean);

            return (mean, variance);
        }

        /// <summary>平均/分散</summary>
        public static (Field mean, Field variance) AverageVariance(Field x, int axis, bool keepdims = false) {
            return AverageVariance(x, new int[] { axis }, keepdims);
        }
    }
}
