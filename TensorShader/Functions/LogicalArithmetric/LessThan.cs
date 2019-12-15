namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>小なり</summary>
        public static VariableNode LessThan(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LessThan(x1.Shape));
        }

        /// <summary>小なり</summary>
        public static VariableNode LessThan(VariableNode x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.GreaterThanLeftConstant(c, x.Shape));
        }

        /// <summary>小なり</summary>
        public static VariableNode LessThan(float c, VariableNode x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.LessThanLeftConstant(c, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>小なり</summary>
        public static Tensor LessThan(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.LessThan(x1.Shape));
        }

        /// <summary>小なり</summary>
        public static Tensor LessThan(Tensor x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.GreaterThanLeftConstant(c, x.Shape));
        }

        /// <summary>小なり</summary>
        public static Tensor LessThan(float c, Tensor x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.LessThanLeftConstant(c, x.Shape));
        }
    }
}
