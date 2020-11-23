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
        public static implicit operator VariableField(float[][] val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator VariableField(float[,,] val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator VariableField(float[][,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator VariableField(float[,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator VariableField(float[][,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator VariableField(float[,,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator VariableField(float[][,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>任意形状の生成</summary>
        public static implicit operator VariableField(Shape shape) { 
            return (Tensor)shape;
        }

        /// <summary>任意形状任意初期値の生成</summary>
        public static implicit operator VariableField((Shape shape, float[] v) val) { 
            return (Tensor)(val.shape, val.v);
        }

        /// <summary>任意形状任意初期値の生成</summary>
        public static implicit operator VariableField((Shape shape, float[][] v) val) { 
            return (Tensor)(val.shape, val.v);
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
        public static implicit operator ParameterField(float[][] val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator ParameterField(float[,,] val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator ParameterField(float[][,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator ParameterField(float[,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator ParameterField(float[][,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator ParameterField(float[,,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator ParameterField(float[][,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>任意形状の生成</summary>
        public static implicit operator ParameterField(Shape shape) { 
            return (Tensor)shape;
        }

        /// <summary>任意形状任意初期値の生成</summary>
        public static implicit operator ParameterField((Shape shape, float[] v) val) { 
            return (Tensor)(val.shape, val.v);
        }

        /// <summary>任意形状任意初期値の生成</summary>
        public static implicit operator ParameterField((Shape shape, float[][] v) val) { 
            return (Tensor)(val.shape, val.v);
        }
    }
}
