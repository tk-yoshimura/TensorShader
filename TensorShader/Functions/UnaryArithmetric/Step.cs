namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>単位ステップ関数</summary>
        public static VariableNode Step(VariableNode x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Step(x.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>単位ステップ関数</summary>
        public static Tensor Step(Tensor x) {
            return UnaryArithmetric(x, new Operators.UnaryArithmetric.Step(x.Shape));
        }
    }
}
