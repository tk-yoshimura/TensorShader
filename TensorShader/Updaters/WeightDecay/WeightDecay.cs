namespace TensorShader.Updaters.WeightDecay {
    /// <summary>重み減衰</summary>
    public abstract class WeightDecay : Updater {
        /// <summary>コンストラクタ</summary>
        public WeightDecay(ParameterField parameter)
            : base(parameter) { }
    }
}
