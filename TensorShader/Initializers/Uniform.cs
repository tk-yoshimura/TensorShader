using System;
using TensorShader.Operators.RandomGeneration;

namespace TensorShader.Initializers {
    /// <summary>一様乱数</summary>
    public class Uniform : Initializer {
        private readonly UniformRandom generator;
        private readonly Flow flow;

        /// <summary>最小値</summary>
        public float Min { private set; get; }

        /// <summary>最大値</summary>
        public float Max { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Uniform(Tensor tensor, Random random)
            : this(tensor, random, 0f, 1f) { }

        /// <summary>コンストラクタ</summary>
        public Uniform(Tensor tensor, Random random, float min = 0f, float max = 1f)
            : base(tensor) {
            this.generator = new UniformRandom(tensor.Shape, random);

            if (min != 0f || max != 1f) {
                InputNode inputnode = tensor;
                VariableNode node = inputnode * (max - min) + min;
                node.Update(inputnode);

                this.flow = Flow.FromInputs(inputnode);
            }
            else {
                this.flow = null;
            }

            this.Min = min;
            this.Max = max;
        }

        /// <summary>コンストラクタ</summary>
        public Uniform(Tensor tensor, Random random, float range = 1f)
            : this(tensor, random, -range, range) { }

        /// <summary>初期化フロー</summary>
        public override void Execute() {
            generator.Execute(Tensor);

            if (flow != null) {
                flow.Execute();
            }
        }
    }
}
