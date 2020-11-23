﻿using System;

namespace TensorShader {
    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> {        

        /// <summary>スカラーの生成</summary>
        public static implicit operator NdimArray<T>(T value) { 
            return new NdimArray<T>(Shape.Scalar, new T[]{ value });
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
                throw new ArgumentException("batch");
            }

            return new NdimArray<T>(val.shape, val.v.Flatten());
        }

        /// <summary>float配列へ変換</summary>
        public static explicit operator T[](NdimArray<T> array) {
            return array.Value;
        }

        /// <summary>floatへ変換</summary>
        /// <exception cref="InvalidCastException">Not Scalar</exception>
        public static explicit operator T(NdimArray<T> array) {
            if(array.Shape != Shape.Scalar) { 
                throw new InvalidCastException();
            }
            
            return array.Value[0];
        }
    }
}