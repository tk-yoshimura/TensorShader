namespace TensorShader {
    /// <summary>変数フィールド</summary>
    public partial class VariableField {
        /// <summary>スカラーの生成</summary>
        public static implicit operator VariableField(float val) { 
            return (Tensor)val;
        }

        /// <summary>ベクターの生成</summary>
        public static implicit operator VariableField(float[] val) { 
            return (Tensor)val;
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator VariableField(float[,] val) { 
            return (Tensor)val;
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator VariableField((float[] v, (int channels, int batch) shape) val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator VariableField(float[,,] val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator VariableField((float[] v, (int channels, int width, int batch) shape) val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator VariableField(float[,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator VariableField((float[] v, (int channels, int width, int height, int batch) shape) val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator VariableField(float[,,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator VariableField((float[] v, (int channels, int width, int height, int depth, int batch) shape) val) { 
            return (Tensor)val;
        }
    }

    /// <summary>パラメータフィールド</summary>
    public partial class ParameterField {
        /// <summary>スカラーの生成</summary>
        public static implicit operator ParameterField(float val) { 
            return (Tensor)val;
        }

        /// <summary>ベクターの生成</summary>
        public static implicit operator ParameterField(float[] val) { 
            return (Tensor)val;
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator ParameterField(float[,] val) { 
            return (Tensor)val;
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator ParameterField((float[] v, (int channels, int batch) shape) val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator ParameterField(float[,,] val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator ParameterField((float[] v, (int channels, int width, int batch) shape) val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator ParameterField(float[,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator ParameterField((float[] v, (int channels, int width, int height, int batch) shape) val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator ParameterField(float[,,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator ParameterField((float[] v, (int channels, int width, int height, int depth, int batch) shape) val) { 
            return (Tensor)val;
        }
    }
}
