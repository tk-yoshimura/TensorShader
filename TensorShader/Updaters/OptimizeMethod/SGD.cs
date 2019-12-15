using System.Collections.Generic;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>確率的勾配降下法</summary>
    public class SGD : OptimizeMethod {
        /// <summary>学習定数</summary>
        protected readonly InputNode lambda;

        /// <summary>学習定数</summary>
        public float Lambda {
            get {
                return lambda.Tensor.State[0];
            }
            set {
                lambda.Tensor.State = new float[] { value };
            }
        }

        /// <summary>コンストラクタ</summary>
        public SGD(ParameterField parameter, float lambda = 0.01f)
            : base(parameter) {
            this.lambda = new InputNode(new Tensor(Shape.Scalar(), new float[] { lambda }));
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_value = Value - lambda * Grad;
            new_value.Update(Value);

            return Flow.FromInputs(Value, Grad, lambda);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "lambda", lambda.Tensor },
                };

                return table;
            }
        }
    }
}
