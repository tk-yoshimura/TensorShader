using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>RMSpropGraves</summary>
    /// <remarks>
    /// Alex Graves
    /// Generating Sequences With Recurrent Neural Networks
    /// arXiv:1308.0850
    /// </remarks>
    public class RMSpropGraves : OptimizeMethod {
        private readonly InputNode m, v;

        /// <summary>学習定数</summary>
        protected readonly InputNode lambda;

        /// <summary>減衰定数</summary>
        protected readonly InputNode rho;

        /// <summary>学習定数</summary>
        public float Lambda {
            get {
                return lambda.Tensor.State[0];
            }
            set {
                lambda.Tensor.State = new float[] { value };
            }
        }

        /// <summary>減衰定数</summary>
        public float Rho {
            get {
                return rho.Tensor.State[0];
            }
            set {
                rho.Tensor.State = new float[] { value };
            }
        }

        /// <summary>ゼロ除算を回避するための微小正数</summary>
        public float Eps { private set; get; }

        /// <summary>コンストラクタ</summary>
        public RMSpropGraves(ParameterField parameter, float lambda = 1e-3f, float rho = 0.9f, float eps = 1e-5f)
            : base(parameter) {
            this.m = new InputNode(new Tensor(parameter.Shape));
            this.v = new InputNode(new Tensor(parameter.Shape));

            this.lambda = new InputNode(new Tensor(Shape.Scalar(), new float[] { lambda }));
            this.rho = new InputNode(new Tensor(Shape.Scalar(), new float[] { rho }));

            this.Eps = eps;

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_m = rho * m + (1 - rho) * Grad;
            VariableNode new_v = rho * v + (1 - rho) * Square(Grad);
            VariableNode new_value = Value - lambda * Grad * Rsqrt(new_v - Square(new_m) + Eps);

            new_m.Update(m);
            new_v.Update(v);
            new_value.Update(Value);

            return Flow.FromInputs(Value, Grad, m, v, lambda, rho);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "m", m.Tensor },
                    { "v", v.Tensor },
                    { "lambda", lambda.Tensor },
                    { "rho", rho.Tensor },
                };

                return table;
            }
        }

        /// <summary>初期化</summary>
        public override void Initialize() {
            m.Tensor.Zeroset();
            v.Tensor.Zeroset();
        }
    }
}
