namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>LogicalXor</summary>
        public static VariableNode LogicalXor(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LogicalXor(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>LogicalXor</summary>
        public static Tensor LogicalXor(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LogicalXor(x1.Shape));
        }
    }
}
