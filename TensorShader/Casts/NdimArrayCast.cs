using System;

namespace TensorShader {
    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> {

        /// <summary>スカラーの生成</summary>
        public static implicit operator NdimArray<T>(T value) {
            return new NdimArray<T>(Shape.Scalar, new T[] { value });
        }

        /// <summary>ベクターの生成</summary>
        public static implicit operator NdimArray<T>(T[] value) {
            return new NdimArray<T>(Shape.Vector(value.Length), value);
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator NdimArray<T>(T[,] val) {
            return new NdimArray<T>(
                Shape.Map0D(val.GetLength(1), val.GetLength(0)),
                val.Flatten()
            );
        }

        /// <summary>0次元マップの生成</summary>
        public static implicit operator NdimArray<T>(T[][] val) {
            return new NdimArray<T>(
                Shape.Map0D(val[0].GetLength(0), val.Length),
                val.Flatten()
            );
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator NdimArray<T>(T[,,] val) {
            return new NdimArray<T>(
                Shape.Map1D(val.GetLength(2), val.GetLength(1), val.GetLength(0)),
                val.Flatten()
            );
        }

        /// <summary>1次元マップの生成</summary>
        public static implicit operator NdimArray<T>(T[][,] val) {
            return new NdimArray<T>(
                Shape.Map1D(val[0].GetLength(1), val[0].GetLength(0), val.Length),
                val.Flatten()
            );
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator NdimArray<T>(T[,,,] val) {
            return new NdimArray<T>(
                Shape.Map2D(val.GetLength(3), val.GetLength(2), val.GetLength(1), val.GetLength(0)),
                val.Flatten()
            );
        }

        /// <summary>2次元マップの生成</summary>
        public static implicit operator NdimArray<T>(T[][,,] val) {
            return new NdimArray<T>(
                Shape.Map2D(val[0].GetLength(2), val[0].GetLength(1), val[0].GetLength(0), val.Length),
                val.Flatten()
            );
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator NdimArray<T>(T[,,,,] val) {
            return new NdimArray<T>(
                Shape.Map3D(val.GetLength(4), val.GetLength(3), val.GetLength(2), val.GetLength(1), val.GetLength(0)),
                val.Flatten()
            );
        }

        /// <summary>3次元マップの生成</summary>
        public static implicit operator NdimArray<T>(T[][,,,] val) {
            return new NdimArray<T>(
                Shape.Map3D(val[0].GetLength(3), val[0].GetLength(2), val[0].GetLength(1), val[0].GetLength(0), val.Length),
                val.Flatten()
            );
        }

        /// <summary>任意形状の生成</summary>
        public static implicit operator NdimArray<T>(Shape shape) {
            return new NdimArray<T>(shape);
        }

        /// <summary>任意形状任意初期値の生成</summary>
        public static implicit operator NdimArray<T>((Shape shape, T[] v) val) {
            return new NdimArray<T>(val.shape, val.v);
        }

        /// <summary>任意形状任意初期値の生成</summary>
        public static implicit operator NdimArray<T>((Shape shape, T[][] v) val) {
            if (val.shape.Batch != val.v.Length) {
                throw new ArgumentException(ExceptionMessage.Argument("Batch", val.v.Length, val.shape.Batch));
            }

            return new NdimArray<T>(val.shape, val.v.Flatten());
        }

        /// <summary>平坦配列へ変換</summary>
        public static implicit operator T[](NdimArray<T> array) {
            return array.Value;
        }

        /// <summary>単要素へ変換</summary>
        /// <exception cref="InvalidCastException">Not Scalar</exception>
        public static implicit operator T(NdimArray<T> array) {
            if (array.Shape != Shape.Scalar) {
                throw new InvalidCastException(ExceptionMessage.ShapeElements(array.Shape, ("Type", ShapeType.Scalar)));
            }

            return array.Value[0];
        }

        /// <summary>2次配列へ変換</summary>
        /// <exception cref="InvalidCastException">Not 2dim</exception>
        public static implicit operator T[,](NdimArray<T> array) {
            if (array.Ndim != 2) {
                throw new InvalidCastException(ExceptionMessage.ShapeElements(array.Shape, ("Ndim", 2)));
            }

            return array.Value.To2DArray((array.Shape[0], array.Shape[1]));
        }

        /// <summary>1次配列配列へ変換</summary>
        /// <exception cref="InvalidCastException">Not 2dim, Not Map</exception>
        public static implicit operator T[][](NdimArray<T> array) {
            if (array.Type != ShapeType.Map) {
                throw new InvalidCastException(ExceptionMessage.ShapeElements(array.Shape, ("Type", ShapeType.Map)));
            }

            return array.Value.To1DArrays(array.Shape.DataSize, array.Batch);
        }

        /// <summary>3次配列へ変換</summary>
        /// <exception cref="InvalidCastException">Not 3dim</exception>
        public static implicit operator T[,,](NdimArray<T> array) {
            if (array.Ndim != 3) {
                throw new InvalidCastException(ExceptionMessage.ShapeElements(array.Shape, ("Ndim", 3)));
            }

            return array.Value.To3DArray((array.Shape[0], array.Shape[1], array.Shape[2]));
        }

        /// <summary>2次配列配列へ変換</summary>
        /// <exception cref="InvalidCastException">Not 3dim, Not Map</exception>
        public static implicit operator T[][,](NdimArray<T> array) {
            if (array.Ndim != 3 || array.Type != ShapeType.Map) {
                throw new InvalidCastException(ExceptionMessage.ShapeElements(array.Shape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            return array.Value.To2DArrays((array.Shape[0], array.Shape[1]), array.Batch);
        }

        /// <summary>4次配列へ変換</summary>
        /// <exception cref="InvalidCastException">Not 4dim</exception>
        public static implicit operator T[,,,](NdimArray<T> array) {
            if (array.Ndim != 4) {
                throw new InvalidCastException(ExceptionMessage.ShapeElements(array.Shape, ("Ndim", 4)));
            }

            return array.Value.To4DArray((array.Shape[0], array.Shape[1], array.Shape[2], array.Shape[3]));
        }

        /// <summary>3次配列配列へ変換</summary>
        /// <exception cref="InvalidCastException">Not 4dim, Not Map</exception>
        public static implicit operator T[][,,](NdimArray<T> array) {
            if (array.Ndim != 4 || array.Type != ShapeType.Map) {
                throw new InvalidCastException(ExceptionMessage.ShapeElements(array.Shape, ("Ndim", 4), ("Type", ShapeType.Map)));
            }

            return array.Value.To3DArrays((array.Shape[0], array.Shape[1], array.Shape[2]), array.Batch);
        }

        /// <summary>5次配列へ変換</summary>
        /// <exception cref="InvalidCastException">Not 5dim</exception>
        public static implicit operator T[,,,,](NdimArray<T> array) {
            if (array.Ndim != 5) {
                throw new InvalidCastException(ExceptionMessage.ShapeElements(array.Shape, ("Ndim", 5)));
            }

            return array.Value.To5DArray((array.Shape[0], array.Shape[1], array.Shape[2], array.Shape[3], array.Shape[4]));
        }

        /// <summary>4次配列配列へ変換</summary>
        /// <exception cref="InvalidCastException">Not 5dim, Not Map</exception>
        public static implicit operator T[][,,,](NdimArray<T> array) {
            if (array.Ndim != 5 || array.Type != ShapeType.Map) {
                throw new InvalidCastException(ExceptionMessage.ShapeElements(array.Shape, ("Ndim", 5), ("Type", ShapeType.Map)));
            }

            return array.Value.To4DArrays((array.Shape[0], array.Shape[1], array.Shape[2], array.Shape[3]), array.Batch);
        }
    }
}
