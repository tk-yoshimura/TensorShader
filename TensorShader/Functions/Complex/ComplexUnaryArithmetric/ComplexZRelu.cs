namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素ZRelu</summary>
        /// <remarks>ガウス平面第1象限以外を0とする</remarks>
        public static VariableNode ComplexZRelu(VariableNode x) {
            return ComplexUnaryArithmetric(x, new Operators.ComplexUnaryArithmetric.ComplexZRelu(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素ZRelu</summary>
        /// <remarks>ガウス平面第1象限以外を0とする</remarks>
        public static Tensor ComplexZRelu(Tensor x) {
            return ComplexUnaryArithmetric(x, new Operators.ComplexUnaryArithmetric.ComplexZRelu(x.Shape));
        }
    }
}
