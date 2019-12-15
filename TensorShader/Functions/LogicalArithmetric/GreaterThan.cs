namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>大なり</summary>
        public static VariableNode GreaterThan(VariableNode x1, VariableNode x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.GreaterThan(x1.Shape));
        }

        /// <summary>大なり</summary>
        public static VariableNode GreaterThan(VariableNode x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.LessThanLeftConstant(c, x.Shape));
        }

        /// <summary>大なり</summary>
        public static VariableNode GreaterThan(float c, VariableNode x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.GreaterThanLeftConstant(c, x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>大なり</summary>
        public static Tensor GreaterThan(Tensor x1, Tensor x2) {
            return BinaryArithmetric(x1, x2, new Operators.LogicalArithmetric.GreaterThan(x1.Shape));
        }

        /// <summary>大なり</summary>
        public static Tensor GreaterThan(Tensor x, float c) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.LessThanLeftConstant(c, x.Shape));
        }

        /// <summary>大なり</summary>
        public static Tensor GreaterThan(float c, Tensor x) {
            return BinaryLeftConstantArithmetric(x, new Operators.LogicalArithmetric.GreaterThanLeftConstant(c, x.Shape));
        }
    }
}
