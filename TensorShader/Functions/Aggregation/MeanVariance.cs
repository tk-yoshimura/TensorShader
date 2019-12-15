namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>平均/分散</summary>
        public static (VariableNode mean, VariableNode variance) AverageVariance(VariableNode x, int[] axes = null, bool keepdims = false) {
            VariableNode mean = Average(x, axes, keepdims);
            VariableNode variance = SquareAverage(x, axes, keepdims) - Square(mean);

            return (mean, variance);
        }
    }

    public partial class Tensor {
        /// <summary>平均/分散</summary>
        public static (Tensor mean, Tensor variance) AverageVariance(Tensor x, int[] axes = null, bool keepdims = false) {
            Tensor mean = Average(x, axes, keepdims);
            Tensor variance = SquareAverage(x, axes, keepdims) - Square(mean);

            return (mean, variance);
        }
    }
}
