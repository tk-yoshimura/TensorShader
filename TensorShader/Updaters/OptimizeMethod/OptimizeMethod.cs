namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>勾配降下法</summary>
    public abstract class OptimizeMethod : Updater {
        /// <summary>コンストラクタ</summary>
        public OptimizeMethod(ParameterField parameter)
            : base(parameter) { }

        /// <summary>カハンの加算アルゴリズム</summary>
        internal static (VariableNode new_x, VariableNode new_c) KahanSum(VariableNode x, VariableNode dx, VariableNode c) {
            VariableNode y = dx - c;
            VariableNode t = x + y;
            VariableNode new_c = (t - x) - y;
            VariableNode new_x = t;

            return (new_x, new_c);
        }
    }
}
