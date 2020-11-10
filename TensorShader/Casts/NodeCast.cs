namespace TensorShader {
    /// <summary>入力ノード</summary>
    public partial class InputNode {
        /// <summary>スカラーの生成</summary>
        public static implicit operator InputNode(float val) { 
            return (Tensor)val;
        }

        /// <summary>ベクターの生成</summary>
        public static implicit operator InputNode(float[] val) { 
            return (Tensor)val;
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator InputNode(float[,] val) { 
            return (Tensor)val;
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator InputNode(((int channels, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator InputNode(float[,,] val) { 
            return (Tensor)val;
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator InputNode(((int channels, int width, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator InputNode(float[,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator InputNode(((int channels, int width, int height, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator InputNode(float[,,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator InputNode(((int channels, int width, int height, int depth, int batch) shape, float[] v) val) { 
            return (Tensor)val;
        }
    }
}
