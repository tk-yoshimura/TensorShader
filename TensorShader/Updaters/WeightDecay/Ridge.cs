using System.Collections.Generic;

namespace TensorShader.Updaters.WeightDecay {
    /// <summary>Ridge減衰</summary>
    public class Ridge : WeightDecay {
        /// <summary>減衰率</summary>
        protected readonly InputNode decay;

        /// <summary>減衰率</summary>
        public float Decay {
            get {
                return decay.State[0];
            }
            set {
                decay.State = new float[] { value };
            }
        }

        /// <summary>コンストラクタ</summary>
        public Ridge(ParameterField parameter, float decay)
            : base(parameter) {
            this.decay = new InputNode(new Tensor(Shape.Scalar, new float[] { decay }));
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_value = Value * (1 - decay);
            new_value.Update(Value);

            return Flow.FromInputs(Value, decay);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "decay", decay.Tensor },
                };

                return table;
            }
        }
    }
}
