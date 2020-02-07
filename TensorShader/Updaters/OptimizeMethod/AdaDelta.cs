using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>AdaDelta</summary>
    /// <remarks>
    /// Matthew D. Zeiler
    /// ADADELTA: An Adaptive Learning Rate Method
    /// arXiv:1212.5701
    /// </remarks>
    public class AdaDelta : OptimizeMethod {
        private readonly InputNode r, v;

        /// <summary>減衰定数</summary>
        protected readonly InputNode rho;

        /// <summary>減衰定数</summary>
        public float Rho {
            get {
                return rho.State[0];
            }
            set {
                rho.State = new float[] { value };
            }
        }

        /// <summary>ゼロ除算を回避するための微小正数</summary>
        public float Eps { private set; get; }

        /// <summary>コンストラクタ</summary>
        public AdaDelta(ParameterField parameter, float rho = 0.95f, float eps = 1e-5f)
            : base(parameter) {
            this.r = new InputNode(new Tensor(parameter.Shape));
            this.v = new InputNode(new Tensor(parameter.Shape));

            this.rho = new InputNode(new Tensor(Shape.Scalar(), new float[] { rho }));

            this.Eps = eps;

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_r = r + (1 - rho) * (Square(Grad) - r);
            VariableNode s = Sqrt((v + Eps) / (new_r + Eps)) * Grad;
            VariableNode new_v = v + (1 - rho) * (Square(s) - v);

            VariableNode new_value = Value - s;

            new_r.Update(r);
            new_v.Update(v);
            new_value.Update(Value);

            return Flow.FromInputs(Value, Grad, r, v, rho);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "r", r.Tensor },
                    { "v", v.Tensor },
                    { "rho", rho.Tensor },
                };

                return table;
            }
        }

        /// <summary>初期化</summary>
        public override void Initialize() {
            r.Tensor.Zeroset();
            v.Tensor.Zeroset();
        }
    }
}
