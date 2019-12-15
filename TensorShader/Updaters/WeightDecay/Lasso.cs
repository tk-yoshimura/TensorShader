using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader.Updaters.WeightDecay {
    /// <summary>Lasso回帰</summary>
    public class Lasso : WeightDecay {
        /// <summary>減衰率</summary>
        protected readonly InputNode decay;

        /// <summary>減衰率</summary>
        public float Decay {
            get {
                return decay.Tensor.State[0];
            }
            set {
                decay.Tensor.State = new float[] { value };
            }
        }

        /// <summary>コンストラクタ</summary>
        public Lasso(ParameterField parameter, float decay)
            : base(parameter) {
            this.decay = new InputNode(new Tensor(Shape.Scalar(), new float[] { decay }));
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_value = SoftThreshold(Value, decay);
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
