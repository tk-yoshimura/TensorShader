namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>Decay勾配</summary>
        public static VariableNode TrivectorDecayGrad(VariableNode v, VariableNode u) {
            return TrivectorBinaryArithmetric(v, u, new Operators.TrivectorBinaryArithmetric.TrivectorDecayGrad(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>Decay勾配</summary>
        public static Tensor TrivectorDecayGrad(Tensor v, Tensor u) {
            return TrivectorBinaryArithmetric(v, u, new Operators.TrivectorBinaryArithmetric.TrivectorDecayGrad(v.Shape));
        }
    }
}
