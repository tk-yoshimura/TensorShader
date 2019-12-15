namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>範囲内クリッピング</summary>
        public static VariableNode Clip(VariableNode x, float cmin, float cmax) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Clip(cmin, cmax, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>範囲内クリッピング</summary>
        public static Tensor Clip(Tensor x, float cmin, float cmax) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Clip(cmin, cmax, x.Shape));
        }
    }
}
