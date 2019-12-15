namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>全象限対応逆正接関数</summary>
        public static VariableNode Arctan2(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Arctan2(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>全象限対応逆正接関数</summary>
        public static Tensor Arctan2(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.Arctan2(x1.Shape));
        }
    }
}
