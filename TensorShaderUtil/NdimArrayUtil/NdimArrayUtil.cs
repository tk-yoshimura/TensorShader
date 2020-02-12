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

        /// <summary>任意の写像を適用</summary>
        public static NdimArray<T> Select<T>(this NdimArray<T> arr, Func<T, T> func) {
            T[] val = new T[arr.Length];
            T[] arrval = arr.Value;

            for (int i = 0; i < arr.Length; i++) {
                val[i] = func(arrval[i]);
            }

            return new NdimArray<T>(val, arr.Shape, clone_value: false);
        }
    }
}
