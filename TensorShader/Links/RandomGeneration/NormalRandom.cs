using System;

namespace TensorShader {
    public partial class Field {
        /// <summary>正規乱数を生成(XorShift, Box-Muller Method)</summary>
        public static VariableField NormalRandom(Shape shape, Random random) {
            return VariableNode.NormalRandom(shape, random);
        }
    }
}
