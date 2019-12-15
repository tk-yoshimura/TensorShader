using System;

namespace TensorShader.Operators.RandomGeneration {
    /// <summary>ベルヌーイ分布に従う2値</summary>
    internal class BinaryRandom : RandomGeneration {
        /// <summary>1となる確率</summary>
        public float Prob { private set; get; }

        /// <summary>コンストラクタ</summary>
        public BinaryRandom(Shape shape, Random random, float prob)
            : base(shape, random) {
            this.Prob = prob;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor refmap = tensors[0];

            TensorShaderCudaBackend.Randomize.Bernoulli((uint)refmap.Length, Prob, refmap.Buffer, Random);
        }
    }
}
