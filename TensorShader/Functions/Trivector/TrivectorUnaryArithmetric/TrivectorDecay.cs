namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトル減衰</summary>
        /// <remarks>(norm/(norm+1))を乗ずる</remarks>
        public static VariableNode TrivectorDecay(VariableNode v) {
            return TrivectorUnaryArithmetric(v, new Operators.TrivectorUnaryArithmetric.TrivectorDecay(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトル減衰</summary>
        /// <remarks>(norm/(norm+1))を乗ずる</remarks>
        public static Tensor TrivectorDecay(Tensor v) {
            return TrivectorUnaryArithmetric(v, new Operators.TrivectorUnaryArithmetric.TrivectorDecay(v.Shape));
        }
    }
}
