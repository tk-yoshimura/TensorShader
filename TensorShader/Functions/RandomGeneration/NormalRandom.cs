using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>正規乱数を生成(XorShift, Box-Muller Method)</summary>
        public static InputNode NormalRandom(Shape shape, Random random) {
            Tensor y = new(shape);

            InputNode inputnode = y;
            inputnode.Initializer = new Initializers.Normal(y, random);

            return inputnode;
        }
    }

    public partial class Tensor {
        /// <summary>正規乱数を生成(XorShift, Box-Muller Method)</summary>
        public static Tensor NormalRandom(Shape shape, Random random) {
            Tensor y = new(shape);

            Operator ope = new Operators.RandomGeneration.NormalRandom(shape, random);

            ope.Execute(y);

            return y;
        }
    }
}
