namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>繰り上げ関数</summary>
        public static VariableNode Ceil(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Ceil(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>繰り上げ関数</summary>
        public static Tensor Ceil(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Ceil(x.Shape));
        }
    }
}
