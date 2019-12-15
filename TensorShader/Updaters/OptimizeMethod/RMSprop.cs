using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>RMSprop</summary>
    /// <remarks>
    /// Geoff Hinton
    /// rmsprop: Divide the gradient by a running average of its recent magnitude
    /// http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    /// </remarks>
    public class RMSprop : OptimizeMethod {
        private readonly InputNode m;

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
        public RMSprop(ParameterField parameter, float lambda = 1e-3f, float rho = 0.9f, float eps = 1e-5f)
            : base(parameter) {
            this.m = new InputNode(new Tensor(parameter.Shape));

            this.lambda = new InputNode(new Tensor(Shape.Scalar(), new float[] { lambda }));
            this.rho = new InputNode(new Tensor(Shape.Scalar(), new float[] { rho }));

            this.Eps = eps;

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_m = rho * m + (1 - rho) * Square(Grad);
            VariableNode new_value = Value - lambda * Grad * Rsqrt(new_m + Eps);

            new_m.Update(m);
            new_value.Update(Value);

            return Flow.FromInputs(Value, Grad, m, lambda, rho);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "m", m.Tensor },
                    { "lambda", lambda.Tensor },
                    { "rho", rho.Tensor },
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
