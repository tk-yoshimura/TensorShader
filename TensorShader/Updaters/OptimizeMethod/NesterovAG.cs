using System.Collections.Generic;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>Nesterovの加速勾配降下法</summary>
    /// <remarks>
    /// Yoshua Bengio, Nicolas Boulanger-Lewandowski, Razvan Pascanu
    /// Advances in Optimizing Recurrent Networks
    /// arXiv:1212.0901
    /// </remarks>
    public class NesterovAG : OptimizeMethod {
        private readonly InputNode m, kahan_c;

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
        public NesterovAG(ParameterField parameter, float lambda = 0.01f, float alpha = 0.9f)
            : base(parameter) {
            this.m = new InputNode(new Tensor(parameter.Shape));

            this.kahan_c = new InputNode(new Tensor(parameter.Shape));

            this.lambda = new InputNode(new Tensor(Shape.Scalar, new float[] { lambda }));
            this.alpha = new InputNode(new Tensor(Shape.Scalar, new float[] { alpha }));

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_m = m * alpha - lambda * Grad;
            VariableNode diff_value = alpha * alpha * new_m - (1 + alpha) * lambda * Grad;
            
            (VariableNode new_value, VariableNode new_kahan_c) = KahanSum(Value, diff_value, kahan_c);

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
