using System.Collections.Generic;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>慣性項つき確率的勾配降下法</summary>
    public class MomentumSGD : OptimizeMethod {
        private readonly InputNode m;

        /// <summary>学習定数</summary>
        protected readonly InputNode lambda;

        /// <summary>慣性係数</summary>
        protected readonly InputNode alpha;

        /// <summary>学習定数</summary>
        public float Lambda {
            get {
                return lambda.State[0];
            }
            set {
                lambda.State = new float[] { value };
            }
        }

        /// <summary>慣性係数</summary>
        public float Alpha {
            get {
                return alpha.State[0];
            }
            set {
                alpha.State = new float[] { value };
            }
        }

        /// <summary>コンストラクタ</summary>
        public MomentumSGD(ParameterField parameter, float lambda = 0.01f, float alpha = 0.9f)
            : base(parameter) {
            this.m = new InputNode(new Tensor(parameter.Shape));
            this.lambda = new InputNode(new Tensor(Shape.Scalar, new float[] { lambda }));
            this.alpha = new InputNode(new Tensor(Shape.Scalar, new float[] { alpha }));

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_m = alpha * m - lambda * Grad;
            VariableNode new_value = Value + new_m;

            new_m.Update(m);
            new_value.Update(Value);

            return Flow.FromInputs(Value, Grad, m, lambda, alpha);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "m", m.Tensor },
                    { "lambda", lambda.Tensor },
                    { "alpha", alpha.Tensor },
                };

                return table;
            }
        }

        /// <summary>初期化</summary>
        public override void Initialize() {
            m.Tensor.Zeroset();
        }
    }
}
