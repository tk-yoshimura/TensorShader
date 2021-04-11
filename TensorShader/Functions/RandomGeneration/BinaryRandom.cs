using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>ベルヌーイ分布に従う2値</summary>
        public static InputNode BinaryRandom(Shape shape, Random random, float prob) {
            Tensor y = new(shape);

            InputNode inputnode = y;
            inputnode.Initializer = new Initializers.Binary(y, random, prob);

            return inputnode;
        }
    }

    public partial class Tensor {
        /// <summary>ベルヌーイ分布に従う2値</summary>
        public static Tensor BinaryRandom(Shape shape, Random random, float prob) {
            Tensor y = new(shape);

            Operator ope = new Operators.RandomGeneration.BinaryRandom(shape, random, prob);

            ope.Execute(y);

            return y;
        }
    }
}
