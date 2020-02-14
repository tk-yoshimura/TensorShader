namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>Clamp</summary>
        public static VariableNode Clamp(VariableNode x, VariableNode xmin, VariableNode xmax) {
            return TrinaryArithmetric(x, xmin, xmax, new Operators.TrinaryArithmetric.Clamp(x.Shape));
        }

        /// <summary>Clamp</summary>
        public static VariableNode Clamp(VariableNode x, float xmin, float xmax) {
            return TrinaryBiConstantArithmetric(x, new Operators.TrinaryArithmetric.ClampConstant(xmin, xmax, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>Clamp</summary>
        public static Tensor Clamp(Tensor x, Tensor xmin, Tensor xmax) {
            return TrinaryArithmetric(x, xmin, xmax, new Operators.TrinaryArithmetric.Clamp(x.Shape));
        }

        /// <summary>Clamp</summary>
        public static Tensor Clamp(Tensor x, float xmin, float xmax) {
            return TrinaryBiConstantArithmetric(x, new Operators.TrinaryArithmetric.ClampConstant(xmin, xmax, x.Shape));
        }
    }
}
