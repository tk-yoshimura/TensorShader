using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using TensorShader;

namespace TensorShaderUtil {

    /// <summary>多次元配列生成ヘルパー</summary>
    public static class NdimArrayUtil {

        /// <summary>等差数列</summary>
        /// <param name="start">始点</param>
        /// <param name="end">終点</param>
        /// <param name="num">要素数</param>
        /// <param name="endpoint">終点を含むか</param>
        public static NdimArray<float> Linspace(float start, float end, int num, bool endpoint = true) {
            float[] val = new float[num];

            double s = (end - start) / (endpoint ? (num - 1) : num);

            for (int i = 0; i < num; i++) {
                val[i] = (float)(start + i * s);
            }

            return new NdimArray<float>(val, Shape.Vector(num), clone_value: false);
        }

        /// <summary>等差数列</summary>
        /// <param name="start">始点</param>
        /// <param name="end">終点</param>
        /// <param name="num">要素数</param>
        /// <param name="endpoint">終点を含むか</param>
        public static NdimArray<double> Linspace(double start, double end, int num, bool endpoint = true) {
            double[] val = new double[num];

            double s = (end - start) / (endpoint ? (num - 1) : num);

            for (int i = 0; i < num; i++) {
                val[i] = start + i * s;
            }

            return new NdimArray<double>(val, Shape.Vector(num), clone_value: false);
        }

        /// <summary>直積</summary>
        /// <param name="arr1">配列1</param>
        /// <param name="arr2">配列2</param>
        /// <param name="type">形状タイプ</param>
        /// <returns>Shape:(2, arr1.Shape[0], arr2.Shape[1], ..., arr2.Shape[0], arr2.Shape[1], ...)</returns>
        public static NdimArray<T> Product<T>(NdimArray<T> arr1, NdimArray<T> arr2, ShapeType type) {
            T[] val = new T[2 * arr1.Length * arr2.Length];
            T[] arr1val = arr1.Value, arr2val = arr2.Value;

            for (int j = 0, idx = 0; j < arr2.Length; j++) {
                for (int i = 0; i < arr1.Length; i++, idx += 2) {
                    val[idx] = arr1val[i];
                    val[idx + 1] = arr2val[j];
                }
            }

            List<int> s = new List<int>();
            s.Add(2);
            s.AddRange((int[])arr1.Shape);
            s.AddRange((int[])arr2.Shape);

            return new NdimArray<T>(val, new Shape(type, s.ToArray()), clone_value: false);
        }

        /// <summary>直積</summary>
        /// <param name="arr1">配列1</param>
        /// <param name="arr2">配列2</param>
        /// <param name="arr3">配列2</param>
        /// <param name="type">形状タイプ</param>
        /// <returns>Shape:(3, arr1.Shape[0], arr2.Shape[1], ..., arr2.Shape[0], arr2.Shape[1], ..., arr3.Shape[0], arr3.Shape[1], ...)</returns>
        public static NdimArray<T> Product<T>(NdimArray<T> arr1, NdimArray<T> arr2, NdimArray<T> arr3, ShapeType type) {
            T[] val = new T[3 * arr1.Length * arr2.Length * arr3.Length];
            T[] arr1val = arr1.Value, arr2val = arr2.Value, arr3val = arr3.Value;

            for (int k = 0, idx = 0; k < arr3.Length; k++) {
                for (int j = 0; j < arr2.Length; j++) {
                    for (int i = 0; i < arr1.Length; i++, idx += 3) {
                        val[idx] = arr1val[i];
                        val[idx + 1] = arr2val[j];
                        val[idx + 2] = arr3val[k];
                    }
                }
            }

            List<int> s = new List<int>();
            s.Add(3);
            s.AddRange((int[])arr1.Shape);
            s.AddRange((int[])arr2.Shape);
            s.AddRange((int[])arr3.Shape);

            return new NdimArray<T>(val, new Shape(type, s.ToArray()), clone_value: false);
        }

        /// <summary>任意の写像を適用</summary>
        public static NdimArray<float> Select(this NdimArray<float> arr, Func<float, float> func) {
            float[] val = new float[arr.Length];
            float[] arrval = arr.Value;

            for (int i = 0; i < arr.Length; i++) {
                val[i] = func(arrval[i]);
            }

            return new NdimArray<float>(val, arr.Shape, clone_value: false);
        }
    }
}
