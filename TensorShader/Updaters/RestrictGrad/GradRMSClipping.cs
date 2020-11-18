using System;
using System.Collections.Generic;
using static TensorShader.VariableNode;

namespace TensorShader.Updaters.RestrictGrad {
    /// <summary>勾配の2乗平均平方根を一定以下に制限</summary>
    public class GradRMSClipping : RestrictGrad {
        /// <summary>制限値</summary>
        protected readonly InputNode limit;

        /// <summary>制限値</summary>
        public float Limit {
            get {
                return (float)Math.Sqrt((float)limit.State);
            }
            set {
                limit.State = value * value;
            }
        }

        /// <summary>コンストラクタ</summary>
        public GradRMSClipping(ParameterField parameter, float limit)
            : base(parameter) {
            this.limit = limit * limit;
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode norm = Average(Square(Grad));
            VariableNode rate = Sqrt(limit / Maximum(limit, norm));

            VariableNode new_grad = Grad * rate;
            new_grad.Update(Grad);

            return Flow.FromInputs(Grad, limit);
        }

        /// <summary>内部状態</summary>
        public override Dictionary<string, Tensor> States {
            get {
                Dictionary<string, Tensor> table = new Dictionary<string, Tensor>(){
                    { "limit", limit.Tensor },
                };

                return table;
            }
        }
    }
}
