using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>一様乱数を生成(XorShift)</summary>
        /// <remarks>値域 : [0, 1)</remarks>
        public static InputNode UniformRandom(Shape shape, Random random) {
            Tensor y = new Tensor(shape);

            InputNode inputnode = y;
            inputnode.Initializer = new Initializers.Uniform(y, random);

            return inputnode;
        }
    }

    public partial class Tensor {
        /// <summary>一様乱数を生成(XorShift)</summary>
        /// <remarks>値域 : [0, 1)</remarks>
        public static Tensor UniformRandom(Shape shape, Random random) {
            Tensor y = new Tensor(shape);

            Operator ope = new Operators.RandomGeneration.UniformRandom(shape, random);

            ope.Execute(y);

            return y;
        }
    }
}
