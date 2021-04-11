using System;
using TensorShader.Operators.RandomGeneration;

namespace TensorShader.Initializers {
    /// <summary>正規乱数</summary>
    public class Normal : Initializer {
        private readonly NormalRandom generator;
        private readonly Flow flow;

        /// <summary>偏差</summary>
        public float Scale { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Normal(Tensor tensor, Random random, float scale = 1f)
            : base(tensor) {
            this.generator = new NormalRandom(tensor.Shape, random);

            if (scale != 1) {
                InputNode inputnode = tensor;
                VariableNode node = inputnode * scale;
                node.Update(inputnode);

                this.flow = Flow.FromInputs(inputnode);
            }
            else {
                this.flow = null;
            }

            this.Scale = scale;
        }

        /// <summary>初期化フロー</summary>
        public override void Execute() {
            generator.Execute(Tensor);

            if (flow is not null) {
                flow.Execute();
            }
        }
    }
}
