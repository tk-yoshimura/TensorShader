namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数共役</summary>
        public static VariableNode QuaternionConjugate(VariableNode x) {
            return QuaternionUnaryArithmetric(x, new Operators.QuaternionUnaryArithmetric.QuaternionConjugate(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数共役</summary>
        public static Tensor QuaternionConjugate(Tensor x) {
            return QuaternionUnaryArithmetric(x, new Operators.QuaternionUnaryArithmetric.QuaternionConjugate(x.Shape));
        }
    }
}
