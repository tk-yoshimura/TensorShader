using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader.Updaters.WeightDecay {
    /// <summary>Lasso回帰</summary>
    public class Lasso : WeightDecay {

        /// <summary>勾配スケールに依存するか</summary>
        protected bool DependGrad { private set; get; }

        /// <summary>減衰率</summary>
        protected readonly InputNode decay;

        /// <summary>減衰率</summary>
        public float Decay {
            get {
                return decay.State;
            }
            set {
                decay.State = value;
            }
        }

        /// <summary>コンストラクタ</summary>
        public Lasso(ParameterField parameter, float decay, bool depend_grad = false)
            : base(parameter) {

            this.DependGrad = depend_grad;
            this.decay = decay;
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            if (DependGrad) {
                VariableNode absmean = Average(Abs(Grad));

                VariableNode new_grad = Grad + Sign(Value) * absmean * decay;
                new_grad.Update(Grad);

                return Flow.FromInputs(Value, Grad, decay);
            }
            else {
                VariableNode new_grad = Grad + Sign(Value) * decay;
                new_grad.Update(Grad);

                return Flow.FromInputs(Value, Grad, decay);
            }
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new() {
                    { "decay", decay.Tensor },
                };

                return table;
            }
        }
    }
}
