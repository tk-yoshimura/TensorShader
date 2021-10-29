using System;

namespace TensorShaderCudaBackend {
    /// <summary>環境設定</summary>
    public static class Environment {
        
        /// <summary>計算精度</summary>
        public enum PrecisionMode { 
            /// <summary>Float精度</summary>
            Float,
            /// <summary>Float-Float精度</summary>
            FloatFloat
        };

        private const string fail_enable_cudnn = 
            "To enable cudnn, the cudnn library exists and the precision must be float.";
        private const string undefined_enum = 
            "Undefined enum value.";

        private static bool cudnn_enabled = false;
        private static PrecisionMode precision = PrecisionMode.FloatFloat;

        /// <summary>Cudnn libraryが存在するか</summary>
        public static bool CudnnExists => API.Cudnn.Exists;

        /// <summary>Cudnn libraryが有効か</summary>
        public static bool CudnnEnabled { 
            set {
                if (value && (!CudnnExists || Precision != PrecisionMode.Float)) {
                    throw new InvalidOperationException(fail_enable_cudnn);
                }

                cudnn_enabled = value;
            }
            get => cudnn_enabled; 
        }


        /// <summary>計算精度</summary>
        public static PrecisionMode Precision {
            set {
                if (value == PrecisionMode.FloatFloat && CudnnEnabled) { 
                    throw new InvalidOperationException(fail_enable_cudnn);
                }
                if (!Enum.IsDefined(value)){
                    throw new ArgumentException(undefined_enum);
                }

                precision = value;
            }
            get => precision;
        }
    }
}
