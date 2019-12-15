using System;

namespace TensorShader.Operators.RandomGeneration {
    /// <summary>一様乱数を生成(XorShift)</summary>
    /// <remarks>値域 : [0, 1)</remarks>
    internal class UniformRandom : RandomGeneration {
        /// <summary>コンストラクタ</summary>
        public UniformRandom(Shape shape, Random random)
            : base(shape, random) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor refmap = tensors[0];

            TensorShaderCudaBackend.Randomize.Uniform((uint)refmap.Length, refmap.Buffer, Random);
        }
    }
}
