using System;

namespace TensorShader {
    public partial class Field {
        /// <summary>ベルヌーイ分布に従う2値</summary>
        public static VariableField BinaryRandom(Shape shape, Random random, float prob) {
            return VariableNode.BinaryRandom(shape, random, prob);
        }
    }
}
