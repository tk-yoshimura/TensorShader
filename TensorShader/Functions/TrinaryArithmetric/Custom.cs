namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>カスタム3項演算</summary>
        public static VariableNode TrinaryArithmetric(VariableNode x1, VariableNode x2, VariableNode x3, string funcname, string funccode) {
            return TrinaryArithmetric(x1, x2, x3, new Operators.TrinaryArithmetric.CustomTrinaryArithmetric(x1.Shape, funcname, funccode));
        }

        /// <summary>カスタム3項演算</summary>
        public static VariableNode TrinaryUniConstantArithmetric(float c, VariableNode x1, VariableNode x2, string funcname, string funccode) {
            return TrinaryUniConstantArithmetric(x1, x2, new Operators.TrinaryArithmetric.CustomTrinaryUniConstantArithmetric(c, x1.Shape, funcname, funccode));
        }

        /// <summary>カスタム3項演算</summary>
        public static VariableNode TrinaryBiConstantArithmetric(float c1, float c2, VariableNode x, string funcname, string funccode) {
            return TrinaryBiConstantArithmetric(x, new Operators.TrinaryArithmetric.CustomTrinaryBiConstantArithmetric(c1, c2, x.Shape, funcname, funccode));
        }
    }

    public partial class Tensor {
        /// <summary>カスタム3項演算</summary>
        public static Tensor TrinaryArithmetric(Tensor x1, Tensor x2, Tensor x3, string funcname, string funccode) {
            return TrinaryArithmetric(x1, x2, x3, new Operators.TrinaryArithmetric.CustomTrinaryArithmetric(x1.Shape, funcname, funccode));
        }

        /// <summary>カスタム3項演算</summary>
        public static Tensor TrinaryUniConstantArithmetric(float c, Tensor x1, Tensor x2, string funcname, string funccode) {
            return TrinaryUniConstantArithmetric(x1, x2, new Operators.TrinaryArithmetric.CustomTrinaryUniConstantArithmetric(c, x1.Shape, funcname, funccode));
        }

        /// <summary>カスタム3項演算</summary>
        public static Tensor TrinaryBiConstantArithmetric(float c1, float c2, Tensor x, string funcname, string funccode) {
            return TrinaryBiConstantArithmetric(x, new Operators.TrinaryArithmetric.CustomTrinaryBiConstantArithmetric(c1, c2, x.Shape, funcname, funccode));
        }
    }
}
