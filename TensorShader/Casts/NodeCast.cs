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

        /// <summary>1次元マップの生成</summary>
        public static implicit operator InputNode(float[,,] val) { 
            return (Tensor)val;
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator InputNode(float[,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator InputNode(float[,,,,] val) { 
            return (Tensor)val;
        }

        /// <summary>任意形状の生成</summary>
        public static implicit operator InputNode(Shape shape) { 
            return new Tensor(shape);
        }

        /// <summary>任意形状任意初期値の生成</summary>
        public static implicit operator InputNode((Shape shape, float[] v) val) { 
            return new Tensor(val.shape, val.v);
        }
    }
}
