namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>符号付き平方根</summary>
        public static VariableNode SignedSqrt(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.SignedSqrt(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>符号付き平方根</summary>
        public static Tensor SignedSqrt(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.SignedSqrt(x.Shape));
        }
    }
}
