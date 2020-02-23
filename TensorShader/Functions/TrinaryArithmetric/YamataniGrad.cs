namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>YamataniŒù”z</summary>
        public static VariableNode YamataniGrad(VariableNode x1, VariableNode x2, float slope) {
            return TrinaryUniConstantArithmetric(x1, x2, new Operators.TrinaryArithmetric.YamataniGrad(slope, x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>YamataniŒù”z</summary>
        public static Tensor YamataniGrad(Tensor x1, Tensor x2, float slope) {
            return TrinaryUniConstantArithmetric(x1, x2, new Operators.TrinaryArithmetric.YamataniGrad(slope, x1.Shape));
        }
    }
}
