namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>SoftPlus</summary>
        public static VariableNode SoftPlus(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.SoftPlus(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>SoftPlus</summary>
        public static Tensor SoftPlus(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.SoftPlus(x.Shape));
        }
    }
}
