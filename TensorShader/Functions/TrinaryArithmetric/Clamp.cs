namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>最大要素</summary>
        public static VariableNode Clamp(VariableNode x, VariableNode xmin, VariableNode xmax) {
            return TrinaryArithmetric(x, xmin, xmax, new Operators.TrinaryArithmetric.Clamp(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>最大要素</summary>
        public static Tensor Clamp(Tensor x, Tensor xmin, Tensor xmax) {
            return TrinaryArithmetric(x, xmin, xmax, new Operators.TrinaryArithmetric.Clamp(x.Shape));
        }
    }
}
