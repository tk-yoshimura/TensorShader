namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>符号付きべき関数</summary>
        public static VariableNode SignedPow(VariableNode x, float alpha) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.SignedPow(alpha, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>符号付きべき関数</summary>
        public static Tensor SignedPow(Tensor x, float alpha) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.SignedPow(alpha, x.Shape));
        }
    }
}
