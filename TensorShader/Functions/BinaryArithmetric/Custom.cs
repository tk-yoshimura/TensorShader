namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>カスタム2項演算</summary>
        public static VariableNode BinaryArithmetric(VariableNode x1, VariableNode x2, string funcname, string funccode) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.CustomBinaryArithmetric(x1.Shape, funcname, funccode));
        }

        /// <summary>カスタム2項演算</summary>
        public static VariableNode BinaryConstantArithmetric(float c, VariableNode x, string funcname, string funccode) {
            return BinaryRightConstantArithmetric(x, new Operators.BinaryArithmetric.CustomBinaryConstantArithmetric(c, x.Shape, funcname, funccode));
        }
    }

    public partial class Tensor {
        /// <summary>カスタム2項演算</summary>
        public static Tensor BinaryArithmetric(Tensor x1, Tensor x2, string funcname, string funccode) {
            return BinaryArithmetric(x1, x2, new Operators.BinaryArithmetric.CustomBinaryArithmetric(x1.Shape, funcname, funccode));
        }

        /// <summary>カスタム2項演算</summary>
        public static Tensor BinaryConstantArithmetric(float c, Tensor x, string funcname, string funccode) {
            return BinaryRightConstantArithmetric(x, new Operators.BinaryArithmetric.CustomBinaryConstantArithmetric(c, x.Shape, funcname, funccode));
        }
    }
}
