namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>Yamatani</summary>
        public static VariableNode Yamatani(VariableNode x1, VariableNode x2, float slope) {
            return TrinaryUniConstantArithmetric(x1, x2, new Operators.TrinaryArithmetric.Yamatani(slope, x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>Yamatani</summary>
        public static Tensor Yamatani(Tensor x1, Tensor x2, float slope) {
            return TrinaryUniConstantArithmetric(x1, x2, new Operators.TrinaryArithmetric.Yamatani(slope, x1.Shape));
        }
    }
}
