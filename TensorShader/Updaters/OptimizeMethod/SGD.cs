using System.Collections.Generic;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>確率的勾配降下法</summary>
    public class SGD : OptimizeMethod {
        private readonly InputNode kahan_c;

        /// <summary>学習定数</summary>
        protected readonly InputNode lambda;

        /// <summary>学習定数</summary>
        public float Lambda {
            get {
                return lambda.State[0];
            }
            set {
                lambda.State = new float[] { value };
            }
        }

        /// <summary>コンストラクタ</summary>
        public SGD(ParameterField parameter, float lambda = 0.01f)
            : base(parameter) {
            this.kahan_c = new InputNode(new Tensor(parameter.Shape));

            this.lambda = new InputNode(new Tensor(Shape.Scalar, new float[] { lambda }));
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode diff_value = - lambda * Grad;

            (VariableNode new_value, VariableNode new_kahan_c) = KahanSum(Value, diff_value, kahan_c);

            new_value.Update(Value);
            new_kahan_c.Update(kahan_c);

            return Flow.FromInputs(Value, Grad, kahan_c, lambda);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "kahan_c", kahan_c.Tensor },
                    { "lambda", lambda.Tensor },
                };

                return table;
            }
        }

        /// <summary>初期化</summary>
        public override void Initialize() {
            kahan_c.Tensor.Zeroset();
        }
    }
}
