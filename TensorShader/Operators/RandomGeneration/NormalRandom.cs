using System;

namespace TensorShader.Operators.RandomGeneration {
    /// <summary>正規乱数を生成(XorShift)</summary>
    /// <remarks>値域 : [0, 1)</remarks>
    internal class NormalRandom : RandomGeneration {
        /// <summary>コンストラクタ</summary>
        public NormalRandom(Shape shape, Random random)
            : base(shape, random) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor refmap = tensors[0];

            TensorShaderCudaBackend.Randomize.Normal((uint)refmap.Length, refmap.Buffer, Random);
        }
    }
}
