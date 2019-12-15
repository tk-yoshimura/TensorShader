namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>非数判定</summary>
        public static VariableNode IsNan(VariableNode x) {
            return UnaryArithmetric(x, new Operators.LogicalArithmetric.IsNan(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>非数判定</summary>
        public static Tensor IsNan(Tensor x) {
            return UnaryArithmetric(x, new Operators.LogicalArithmetric.IsNan(x.Shape));
        }
    }
}
