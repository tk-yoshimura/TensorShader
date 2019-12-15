namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトルSquash勾配</summary>
        public static VariableNode TrivectorSquashGrad(VariableNode x1, VariableNode x2) {
            return TrivectorBinaryArithmetric(x1, x2, new Operators.TrivectorBinaryArithmetric.TrivectorSquashGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトルSquash勾配</summary>
        public static Tensor TrivectorSquashGrad(Tensor x1, Tensor x2) {
            return TrivectorBinaryArithmetric(x1, x2, new Operators.TrivectorBinaryArithmetric.TrivectorSquashGrad(x1.Shape));
        }
    }
}
