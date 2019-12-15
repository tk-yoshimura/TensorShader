namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>外積</summary>
        public static VariableNode TrivectorCross(VariableNode v, VariableNode u) {
            return TrivectorBinaryArithmetric(v, u, new Operators.TrivectorBinaryArithmetric.TrivectorCross(v.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>外積</summary>
        public static Tensor TrivectorCross(Tensor v, Tensor u) {
            return TrivectorBinaryArithmetric(v, u, new Operators.TrivectorBinaryArithmetric.TrivectorCross(v.Shape));
        }
    }
}
