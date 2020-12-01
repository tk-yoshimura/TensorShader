using System.Collections.Generic;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>慣性項つき確率的勾配降下法</summary>
    public class MomentumSGD : OptimizeMethod {
        private readonly InputNode m, kahan_c;

        /// <summary>学習定数</summary>
        protected readonly InputNode lambda;

        /// <summary>慣性係数</summary>
        protected readonly InputNode alpha;

        /// <summary>学習定数</summary>
        public float Lambda {
            get {
                return lambda.State;
            }
            set {
                lambda.State = value;
            }
        }

        /// <summary>慣性係数</summary>
        public float Alpha {
            get {
                return alpha.State;
            }
            set {
                alpha.State = value;
            }
        }

        /// <summary>コンストラクタ</summary>
        public MomentumSGD(ParameterField parameter, float lambda = 0.01f, float alpha = 0.9f)
            : base(parameter) {
            this.m = parameter.Shape;

            this.kahan_c = parameter.Shape;

            this.lambda = lambda;
            this.alpha = alpha;

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_m = alpha * m - lambda * Grad;

            (VariableNode new_value, VariableNode new_kahan_c) = KahanSum(Value, new_m, kahan_c);

            new_m.Update(m);
            new_value.Update(Value);
            new_kahan_c.Update(kahan_c);

            return Flow.FromInputs(Value, Grad, m, kahan_c, lambda, alpha);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "m", m.Tensor },
                    { "kahan_c", kahan_c.Tensor },
                    { "lambda", lambda.Tensor },
                    { "alpha", alpha.Tensor },
                };

                return table;
            }
        }

        /// <summary>初期化</summary>
        public override void Initialize() {
            m.Tensor.Zeroset();
            kahan_c.Tensor.Zeroset();
        }
    }
}
