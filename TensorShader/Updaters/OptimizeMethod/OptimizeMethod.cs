namespace TensorShader.Updaters.OptimizeMethod {
    /// <summary>勾配降下法</summary>
    public abstract class OptimizeMethod : Updater {
        /// <summary>コンストラクタ</summary>
        public OptimizeMethod(ParameterField parameter)
            : base(parameter) { }
    }
}
