using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>Adamax</summary>
    /// <remarks>
    /// Kingma, Diederik, and Jimmy Ba.
    /// Adam: A Method for Stochastic Optimization
    /// arXiv:1412.6980v8
    /// </remarks>
    public class Adamax : OptimizeMethod {
        private readonly InputNode m, v, t;

        /// <summary>α</summary>
        protected readonly InputNode alpha;

        /// <summary>β1, β2</summary>
        protected readonly InputNode beta1, beta2;

        /// <summary>α</summary>
        public float Alpha {
            get {
                return alpha.Tensor.State[0];
            }
            set {
                alpha.Tensor.State = new float[] { value };
            }
        }

        /// <summary>β1</summary>
        public float Beta1 {
            get {
                return beta1.Tensor.State[0];
            }
            set {
                beta1.Tensor.State = new float[] { value };
            }
        }

        /// <summary>β2</summary>
        public float Beta2 {
            get {
                return beta2.Tensor.State[0];
            }
            set {
                beta2.Tensor.State = new float[] { value };
            }
        }

        /// <summary>ゼロ除算を回避するための微小正数</summary>
        public float Eps { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Adamax(ParameterField parameter, float alpha = 2e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-5f)
            : base(parameter) {
            this.m = new InputNode(new Tensor(parameter.Shape));
            this.v = new InputNode(new Tensor(parameter.Shape));

            this.t = new InputNode(new Tensor(Shape.Scalar()));

            this.alpha = new InputNode(new Tensor(Shape.Scalar(), new float[] { alpha }));
            this.beta1 = new InputNode(new Tensor(Shape.Scalar(), new float[] { beta1 }));
            this.beta2 = new InputNode(new Tensor(Shape.Scalar(), new float[] { beta2 }));

            this.Eps = eps;

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_t = t + 1;

            VariableNode new_m = beta1 * m + (1 - beta1) * Grad;
            VariableNode new_v = Maximum(beta2 * v, Abs(Grad));

            VariableNode m_hat = new_m / (1 - Pow(beta1, new_t));

            VariableNode new_value = Value - alpha * m_hat / (new_v + Eps);

            new_m.Update(m);
            new_v.Update(v);

            new_t.Update(t);

            new_value.Update(Value);

            return Flow.FromInputs(Value, Grad, m, v, t, alpha, beta1, beta2);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "m", m.Tensor },
                    { "v", v.Tensor },
                    { "t", t.Tensor },
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
        }
    }
}
