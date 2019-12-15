namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトル正規化勾配</summary>
        public static VariableNode TrivectorNormalizeGrad(VariableNode x1, VariableNode x2) {
            return TrivectorBinaryArithmetric(x1, x2, new Operators.TrivectorBinaryArithmetric.TrivectorNormalizeGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトル正規化勾配</summary>
        public static Tensor TrivectorNormalizeGrad(Tensor x1, Tensor x2) {
            return TrivectorBinaryArithmetric(x1, x2, new Operators.TrivectorBinaryArithmetric.TrivectorNormalizeGrad(x1.Shape));
        }
    }
}
