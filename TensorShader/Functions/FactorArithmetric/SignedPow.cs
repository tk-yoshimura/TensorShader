namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>符号付きべき関数</summary>
        public static VariableNode SignedPow(VariableNode x, VariableNode alpha) {
            return FactorArithmetric(x, alpha, new Operators.FactorArithmetric.SignedPow(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>符号付きべき関数</summary>
        public static Tensor SignedPow(Tensor x, Tensor alpha) {
            return FactorArithmetric(x, alpha, new Operators.FactorArithmetric.SignedPow(x.Shape));
        }
    }
}
