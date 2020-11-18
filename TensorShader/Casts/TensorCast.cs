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

        /// <summary>1次元マップの生成</summary>
        public static implicit operator Tensor(float[,,] val) { 
            return new Tensor(
                Shape.Map1D(val.GetLength(2), val.GetLength(1), val.GetLength(0)), 
                val.Cast<float>().ToArray()
            );
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator Tensor(float[,,,] val) { 
            return new Tensor(
                Shape.Map2D(val.GetLength(3), val.GetLength(2), val.GetLength(1), val.GetLength(0)), 
                val.Cast<float>().ToArray()
            );
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator Tensor(float[,,,,] val) { 
            return new Tensor(
                Shape.Map3D(val.GetLength(4), val.GetLength(3), val.GetLength(2), val.GetLength(1), val.GetLength(0)), 
                val.Cast<float>().ToArray()
            );
        }

        /// <summary>任意形状の生成</summary>
        public static implicit operator Tensor(Shape shape) { 
            return new Tensor(shape);
        }

        /// <summary>任意形状任意初期値の生成</summary>
        public static implicit operator Tensor((Shape shape, float[] v) val) { 
            return new Tensor(val.shape, val.v);
        }

        /// <summary>NdimArrayへ変換</summary>
        public static implicit operator NdimArray<float>(Tensor tensor) {
            return tensor.State;
        }

        /// <summary>NdimArrayから変換</summary>
        public static implicit operator Tensor(NdimArray<float> array) {
            return new Tensor(array.Shape, array.Value);
        }
    }
}
