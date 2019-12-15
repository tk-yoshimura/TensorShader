namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>�r���I�_���a</summary>
        public static VariableNode LogicalXor(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LogicalXor(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>�r���I�_���a</summary>
        public static Tensor LogicalXor(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LogicalXor(x1.Shape));
        }
    }
}
