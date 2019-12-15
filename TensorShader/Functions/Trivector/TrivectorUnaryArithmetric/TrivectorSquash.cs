namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトルに1/(1+sqrt(norm))を乗ずる</summary>
        public static VariableNode TrivectorSquash(VariableNode v) {
            return TrivectorUnaryArithmetric(v, new Operators.TrivectorUnaryArithmetric.TrivectorSquash(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトルに1/(1+sqrt(norm))を乗ずる</summary>
        public static Tensor TrivectorSquash(Tensor v) {
            return TrivectorUnaryArithmetric(v, new Operators.TrivectorUnaryArithmetric.TrivectorSquash(v.Shape));
        }
    }
}
