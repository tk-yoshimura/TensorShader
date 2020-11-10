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
        public static implicit operator VariableField(((int channels, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator VariableField(float[,,] val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator VariableField(((int channels, int width, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator VariableField(float[,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator VariableField(((int channels, int width, int height, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator VariableField(float[,,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator VariableField(((int channels, int width, int height, int depth, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>初期フィールドの生成</summary>
        public static implicit operator VariableField(Shape shape) { 
            return new Tensor(shape);
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
        public static implicit operator ParameterField(((int channels, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator ParameterField(float[,,] val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator ParameterField(((int channels, int width, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator ParameterField(float[,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator ParameterField(((int channels, int width, int height, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator ParameterField(float[,,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator ParameterField(((int channels, int width, int height, int depth, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>初期フィールドの生成</summary>
        public static implicit operator ParameterField(Shape shape) { 
            return new Tensor(shape);
        }
    }
}
