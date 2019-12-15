namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>インデクス</summary>
        public static InputNode Index(Shape shape, int axis) {
            Tensor y = new Tensor(shape);

            InputNode inputnode = y;
            inputnode.Initializer = new Initializers.Index(y, axis);

            return inputnode;
        }
    }

    public partial class Tensor {
        /// <summary>インデクス</summary>
        public static Tensor Index(Shape shape, int axis) {
            Tensor y = new Tensor(shape);

            Operator ope = new Operators.ArrayManipulation.Index(shape, axis);

            ope.Execute(y);

            return y;
        }
    }
}
