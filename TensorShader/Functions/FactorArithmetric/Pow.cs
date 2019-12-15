namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>べき関数</summary>
        public static VariableNode Pow(VariableNode x, VariableNode alpha) {
            return FactorArithmetric(x, alpha, new Operators.FactorArithmetric.Pow(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>べき関数</summary>
        public static Tensor Pow(Tensor x, Tensor alpha) {
            return FactorArithmetric(x, alpha, new Operators.FactorArithmetric.Pow(x.Shape));
        }
    }
}
