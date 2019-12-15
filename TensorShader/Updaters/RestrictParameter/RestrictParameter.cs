namespace TensorShader.Updaters.RestrictParameter {
    /// <summary>パラメータを制限</summary>
    public abstract class RestrictParameter : Updater {
        /// <summary>コンストラクタ</summary>
        public RestrictParameter(ParameterField parameter)
            : base(parameter) { }
    }
}
