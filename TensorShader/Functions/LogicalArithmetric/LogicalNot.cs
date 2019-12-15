namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>論理否定</summary>
        public static VariableNode LogicalNot(VariableNode x) {
            return UnaryArithmetric(x, new Operators.LogicalArithmetric.LogicalNot(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>論理否定</summary>
        public static Tensor LogicalNot(Tensor x) {
            return UnaryArithmetric(x, new Operators.LogicalArithmetric.LogicalNot(x.Shape));
        }
    }
}
