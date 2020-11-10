using System.Linq;

namespace TensorShader {
    /// <summary>テンソルクラス</summary>
    public partial class Tensor {        
        /// <summary>スカラーの生成</summary>
        public static implicit operator Tensor(float val) { 
            return new Tensor(Shape.Scalar, new float[]{ val });
        }

        /// <summary>ベクターの生成</summary>
        public static implicit operator Tensor(float[] val) { 
            return new Tensor(Shape.Vector(val.Length), val);
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator Tensor(float[,] val) { 
            return new Tensor(
                Shape.Map0D(val.GetLength(1), val.GetLength(0)), 
                val.Cast<float>().ToArray()
            );
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator Tensor(((int channels, int batch) shape, float[] v) val) { 
            return new Tensor(
                Shape.Map0D(val.shape.channels, val.shape.batch),
                val.v
            );
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator Tensor(float[,,] val) { 
            return new Tensor(
                Shape.Map1D(val.GetLength(2), val.GetLength(1), val.GetLength(0)), 
                val.Cast<float>().ToArray()
            );
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator Tensor(((int channels, int width, int batch) shape, float[] v) val) { 
            return new Tensor(
                Shape.Map1D(val.shape.channels, val.shape.width, val.shape.batch),
                val.v
            );
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator Tensor(float[,,,] val) { 
            return new Tensor(
                Shape.Map2D(val.GetLength(3), val.GetLength(2), val.GetLength(1), val.GetLength(0)), 
                val.Cast<float>().ToArray()
            );
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator Tensor(((int channels, int width, int height, int batch) shape, float[] v) val) { 
            return new Tensor(
                Shape.Map2D(val.shape.channels, val.shape.width, val.shape.height, val.shape.batch),
                val.v
            );
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator Tensor(float[,,,,] val) { 
            return new Tensor(
                Shape.Map3D(val.GetLength(4), val.GetLength(3), val.GetLength(2), val.GetLength(1), val.GetLength(0)), 
                val.Cast<float>().ToArray()
            );
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator Tensor(((int channels, int width, int height, int depth, int batch) shape, float[] v) val) { 
            return new Tensor(
                Shape.Map3D(val.shape.channels, val.shape.width, val.shape.height, val.shape.depth, val.shape.batch),
                val.v
            );
        }
    }
}
