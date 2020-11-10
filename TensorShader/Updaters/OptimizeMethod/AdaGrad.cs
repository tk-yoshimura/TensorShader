using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>AdaGrad</summary>
    /// <remarks>
    /// John Duchi, Elad Hazan, Yoram Singer
    /// Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
    /// http://jmlr.org/papers/v12/duchi11a.html
    /// </remarks>
    public class AdaGrad : OptimizeMethod {
        private readonly InputNode v, kahan_c;

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

        /// <summary>ゼロ除算を回避するための微小正数</summary>
        public float Eps { private set; get; }

        /// <summary>コンストラクタ</summary>
        public AdaGrad(ParameterField parameter, float lambda = 0.01f, float eps = 1e-5f)
            : base(parameter) {
            this.v = parameter.Shape;
            this.kahan_c = parameter.Shape;

            this.lambda = lambda;

            this.Eps = eps;

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_v = v + Square(Grad);
            VariableNode diff_value = -lambda * Grad / (Sqrt(new_v) + Eps);

            (VariableNode new_value, VariableNode new_kahan_c) = KahanSum(Value, diff_value, kahan_c);

            new_v.Update(v);
            new_value.Update(Value);
            new_kahan_c.Update(kahan_c);

            return Flow.FromInputs(Value, Grad, v, kahan_c, lambda);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "v", v.Tensor },
                    { "kahan_c", kahan_c.Tensor },
                    { "lambda", lambda.Tensor },
                };

                return table;
            }
        }

        /// <summary>初期化</summary>
        public override void Initialize() {
            v.Tensor.Zeroset();
            kahan_c.Tensor.Zeroset();
        }
    }
}
