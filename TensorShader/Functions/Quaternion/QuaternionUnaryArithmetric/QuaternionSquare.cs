namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>四元数2乗</summary>
        public static VariableNode QuaternionSquare(VariableNode x) {
            return QuaternionUnaryArithmetric(x, new Operators.QuaternionUnaryArithmetric.QuaternionSquare(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>四元数2乗</summary>
        public static Tensor QuaternionSquare(Tensor x) {
            return QuaternionUnaryArithmetric(x, new Operators.QuaternionUnaryArithmetric.QuaternionSquare(x.Shape));
        }
    }
}
