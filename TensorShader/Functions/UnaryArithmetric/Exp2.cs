namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2乗冪</summary>
        public static VariableNode Exp2(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Exp2(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>2乗冪</summary>
        public static Tensor Exp2(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Exp2(x.Shape));
        }
    }
}
