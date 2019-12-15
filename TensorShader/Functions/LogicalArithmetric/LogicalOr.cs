namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>論理和</summary>
        /// <remarks>Lotfi A.Zadehのファジー論理和に相当</remarks>
        public static VariableNode LogicalOr(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LogicalOr(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>論理和</summary>
        /// <remarks>Lotfi A.Zadehのファジー論理和に相当</remarks>
        public static Tensor LogicalOr(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LogicalOr(x1.Shape));
        }
    }
}
