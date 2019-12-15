using static TensorShader.VariableNode;

namespace TensorShader.Updaters.RestrictParameter {
    /// <summary>ゼロ平均化</summary>
    public class ZeroMeanShift : RestrictParameter {
        private readonly int[] axes;

        /// <summary>平均対象軸</summary>
        public int[] Axes => (int[])axes.Clone();

        /// <summary>コンストラクタ</summary>
        public ZeroMeanShift(ParameterField parameter, int[] axes)
            : base(parameter) {
            this.axes = (int[])axes.Clone();
        }

        /// <summary>更新フロー</summary>
        public override Flow UpdateFlow() {
            VariableNode mean = Broadcast(Average(Value, axes, keepdims: true), Value.Shape);
            VariableNode new_value = Value - mean;
            new_value.Update(Value);

            return Flow.FromInputs(Value);
        }
    }
}
