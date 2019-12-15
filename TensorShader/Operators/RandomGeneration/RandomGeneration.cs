using System;
using System.Collections.Generic;

namespace TensorShader.Operators.RandomGeneration {
    /// <summary>乱数生成(XorShift96)</summary>
    internal abstract class RandomGeneration : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        protected Random Random { private set; get; }

        /// <summary>コンストラクタ</summary>
        public RandomGeneration(Shape shape, Random random) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.Reference, shape),
            };

            this.Shape = shape;
            this.Random = random;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor refmap) {
            Execute(new Tensor[] { refmap });
        }
    }
}
