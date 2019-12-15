using static TensorShader.VariableNode;

namespace TensorShader.Updaters.RestrictGrad {
    /// <summary>勾配の要素ごとの絶対値を一定以下に制限</summary>
    public class GradClipping : RestrictGrad {
        /// <summary>制限値</summary>
        protected float Limit { private set; get; }

        /// <summary>コンストラクタ</summary>
        public GradClipping(ParameterField parameter, float limit)
            : base(parameter) {
            this.Limit = limit;
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode new_grad = Clip(Grad, -Limit, Limit);
            new_grad.Update(Grad);

            return Flow.FromInputs(Grad);
        }
    }
}
