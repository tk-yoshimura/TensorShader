namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数正規化</summary>
        public static VariableNode QuaternionNormalize(VariableNode v) {
            return QuaternionUnaryArithmetric(v, new Operators.QuaternionUnaryArithmetric.QuaternionNormalize(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数正規化</summary>
        public static Tensor QuaternionNormalize(Tensor v) {
            return QuaternionUnaryArithmetric(v, new Operators.QuaternionUnaryArithmetric.QuaternionNormalize(v.Shape));
        }
    }
}
