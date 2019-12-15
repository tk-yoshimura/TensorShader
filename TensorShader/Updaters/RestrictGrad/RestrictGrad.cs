namespace TensorShader.Updaters.RestrictGrad {
    /// <summary>勾配を制限</summary>
    public abstract class RestrictGrad : Updater {
        /// <summary>コンストラクタ</summary>
        public RestrictGrad(ParameterField parameter)
            : base(parameter) { }
    }
}
