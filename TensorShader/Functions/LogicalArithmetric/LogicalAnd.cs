namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>論理積</summary>
        /// <remarks>Lotfi A.Zadehのファジー論理積に相当</remarks>
        public static VariableNode LogicalAnd(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LogicalAnd(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>論理積</summary>
        /// <remarks>Lotfi A.Zadehのファジー論理積に相当</remarks>
        public static Tensor LogicalAnd(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LogicalAnd(x1.Shape));
        }
    }
}
