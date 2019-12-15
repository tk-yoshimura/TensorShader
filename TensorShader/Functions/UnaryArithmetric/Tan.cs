namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>正接関数</summary>
        public static VariableNode Tan(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Tan(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>正接関数</summary>
        public static Tensor Tan(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Tan(x.Shape));
        }
    }
}
