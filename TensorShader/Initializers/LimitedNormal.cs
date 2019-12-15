using System;
using TensorShader.Operators.RandomGeneration;
using static TensorShader.VariableNode;

namespace TensorShader.Initializers {
    /// <summary>制限付き正規乱数</summary>
    public class LimitedNormal : Initializer {
        private readonly NormalRandom generator;
        private readonly Flow flow;

        /// <summary>偏差</summary>
        public float Scale { private set; get; }

        /// <summary>最大偏差</summary>
        public float LimitSigma { private set; get; }

        /// <summary>コンストラクタ</summary>
        public LimitedNormal(Tensor tensor, Random random, float scale, float limitsigma)
            : base(tensor) {
            this.generator = new NormalRandom(tensor.Shape, random);

            InputNode inputnode = tensor;
            VariableNode node = Clip(inputnode, -limitsigma, limitsigma) * scale;
            node.Update(inputnode);

            this.flow = Flow.FromInputs(inputnode);

            this.Scale = scale;
            this.LimitSigma = limitsigma;
        }

        /// <summary>初期化フロー</summary>
        public override void Execute() {
            generator.Execute(Tensor);
            flow.Execute();
        }
    }
}
