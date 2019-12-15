namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3乗</summary>
        public static VariableNode Cube(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Cube(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>3乗</summary>
        public static Tensor Cube(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Cube(x.Shape));
        }
    }
}
