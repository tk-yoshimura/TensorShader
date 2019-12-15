using System;

namespace TensorShader {
    public partial class Field {
        /// <summary>一様乱数を生成(XorShift)</summary>
        /// <remarks>値域 : [0, 1)</remarks>
        public static VariableField UniformRandom(Shape shape, Random random) {
            return VariableNode.UniformRandom(shape, random);
        }
    }
}
