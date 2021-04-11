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
        private readonly InputNode r, v, kahan_c;

        /// <summary>減衰定数</summary>
        protected readonly InputNode rho;

        /// <summary>減衰定数</summary>
        public float Rho {
            get {
                return rho.State;
            }
            set {
                rho.State = value;
            }
        }

        /// <summary>ゼロ除算を回避するための微小正数</summary>
        public float Eps { private set; get; }

        /// <summary>コンストラクタ</summary>
        public AdaDelta(ParameterField parameter, float rho = 0.95f, float eps = 1e-5f)
            : base(parameter) {
            this.r = parameter.Shape;
            this.v = parameter.Shape;

            this.kahan_c = parameter.Shape;

            this.rho = rho;

            this.Eps = eps;

            Initialize();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_r = r + (1 - rho) * (Square(Grad) - r);
            VariableNode s = Sqrt((v + Eps) / (new_r + Eps)) * Grad;
            VariableNode new_v = v + (1 - rho) * (Square(s) - v);

            (VariableNode new_value, VariableNode new_kahan_c) = KahanSum(Value, -s, kahan_c);

            new_r.Update(r);
            new_v.Update(v);
            new_value.Update(Value);
            new_kahan_c.Update(kahan_c);

            return Flow.FromInputs(Value, Grad, r, v, kahan_c, rho);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new() {
                    { "r", r.Tensor },
                    { "v", v.Tensor },
                    { "kahan_c", kahan_c.Tensor },
                    { "rho", rho.Tensor },
                };

                return table;
            }
        }

        /// <summary>初期化</summary>
        public override void Initialize() {
            r.Tensor.Zeroset();
            v.Tensor.Zeroset();
            kahan_c.Tensor.Zeroset();
        }
    }
}
