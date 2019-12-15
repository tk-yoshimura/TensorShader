using System;
using TensorShader.Operators.RandomGeneration;

namespace TensorShader.Initializers {
    /// <summary>ベルヌーイ分布に従う2値</summary>
    public class Binary : Initializer {
        private readonly BinaryRandom generator;

        /// <summary>1となる確率</summary>
        public float Prob => generator.Prob;

        /// <summary>コンストラクタ</summary>
        public Binary(Tensor tensor, Random random, float prob)
            : base(tensor) {
            this.generator = new BinaryRandom(tensor.Shape, random, prob);
        }

        /// <summary>初期化フロー</summary>
        public override void Execute() {
            generator.Execute(Tensor);
        }
    }
}
