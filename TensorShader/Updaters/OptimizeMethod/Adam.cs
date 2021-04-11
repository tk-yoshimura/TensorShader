using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>Adam</summary>
    /// <remarks>
    /// Kingma, Diederik, and Jimmy Ba.
    /// Adam: A method for stochastic optimization.
    /// arXiv:1412.6980 (2014).
    /// </remarks>
    public class Adam : OptimizeMethod {
        private readonly InputNode m, v, t, kahan_c;

        /// <summary>α</summary>
        protected readonly InputNode alpha;

        /// <summary>β1, β2</summary>
        protected readonly InputNode beta1, beta2;

        /// <summary>α</summary>
        public float Alpha {
            get {
                return alpha.State;
            }
            set {
                alpha.State = value;
            }
        }

        /// <summary>β1</summary>
        public float Beta1 {
            get {
                return beta1.State;
            }
            set {
                beta1.State = value;
            }
        }

        /// <summary>β2</summary>
        public float Beta2 {
            get {
                return beta2.State;
            }
            set {
                beta2.State = value;
            }
        }

        /// <summary>ゼロ除算を回避するための微小正数</summary>
        public float Eps { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Adam(ParameterField parameter, float alpha = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-5f)
            : base(parameter) {
            this.m = parameter.Shape;
            this.v = parameter.Shape;

            this.t = Shape.Scalar;

            this.kahan_c = parameter.Shape;

            this.alpha = alpha;
            this.beta1 = beta1;
            this.beta2 = beta2;

            this.Eps = eps;

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_t = t + 1;

            VariableNode new_m = beta1 * m + (1 - beta1) * Grad;
            VariableNode new_v = beta2 * v + (1 - beta2) * Square(Grad);

            VariableNode m_hat = new_m / (1 - Pow(beta1, new_t));
            VariableNode v_hat = new_v / (1 - Pow(beta2, new_t));

            VariableNode diff_value = -alpha * m_hat * Rsqrt(v_hat + Eps);

            (VariableNode new_value, VariableNode new_kahan_c) = KahanSum(Value, diff_value, kahan_c);

            new_m.Update(m);
            new_v.Update(v);

            new_t.Update(t);

            new_value.Update(Value);
            new_kahan_c.Update(kahan_c);

            return Flow.FromInputs(Value, Grad, m, v, t, kahan_c, alpha, beta1, beta2);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new() {
                    { "m", m.Tensor },
                    { "v", v.Tensor },
                    { "t", t.Tensor },
                    { "kahan_c", kahan_c.Tensor },
                    { "alpha", alpha.Tensor },
                    { "beta1", beta1.Tensor },
                    { "beta2", beta2.Tensor },
                };

                return table;
            }
        }

        /// <summary>初期化</summary>
        public override void Initialize() {
            m.Tensor.Zeroset();
            v.Tensor.Zeroset();
            t.Tensor.Zeroset();
            kahan_c.Tensor.Zeroset();
        }
    }
}
