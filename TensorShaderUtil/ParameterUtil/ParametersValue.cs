using TensorShader;

namespace TensorShaderUtil.ParameterUtil {

    /// <summary>パラメータコンテナの値の一括管理クラス</summary>
    public class ParametersValue<T> {
        private readonly Parameters parameters;
        private readonly string value_name;
        
        /// <summary>コンストラクタ</summary>
        /// <param name="parameters">パラメータコンテナ</param>
        /// <param name="value_name">値識別名(クラス名.プロパティ名)</param>
        public ParametersValue(Parameters parameters, string value_name) { 
            this.parameters = parameters;
            this.value_name = value_name; 
        }

        /// <summary>値</summary>
        public T Value {
            get {
                return (T)parameters[value_name];
            }

            set {
                parameters[value_name] = value;
            }
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return $"{value_name}: {Value}";
        }
    }
}
