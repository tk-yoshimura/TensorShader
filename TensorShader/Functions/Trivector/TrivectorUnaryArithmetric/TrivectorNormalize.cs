namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトル正規化</summary>
        public static VariableNode TrivectorNormalize(VariableNode v) {
            return TrivectorUnaryArithmetric(v, new Operators.TrivectorUnaryArithmetric.TrivectorNormalize(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトル正規化</summary>
        public static Tensor TrivectorNormalize(Tensor v) {
            return TrivectorUnaryArithmetric(v, new Operators.TrivectorUnaryArithmetric.TrivectorNormalize(v.Shape));
        }
    }
}
