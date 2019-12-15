namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2乗</summary>
        public static VariableNode Square(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Square(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>2乗</summary>
        public static Tensor Square(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Square(x.Shape));
        }
    }
}
