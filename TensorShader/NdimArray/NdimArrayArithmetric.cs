using System;

namespace TensorShader {

    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> {

        /// <summary>単項プラス</summary>
        public static NdimArray<T> operator +(NdimArray<T> a) {
            return a;
        }

        /// <summary>単項マイナス</summary>
        public static NdimArray<T> operator -(NdimArray<T> a) {
            NdimArray<T> c = new NdimArray<T>(a.Shape);

            if (c.Value is int[]) {
                int[] av = a.Value as int[], cv = c.Value as int[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = -av[i];
                }

                return c;
            }
            if (c.Value is long[]) {
                long[] av = a.Value as long[], cv = c.Value as long[];

                for (long i = 0; i < cv.Length; i++) {
                    cv[i] = -av[i];
                }

                return c;
            }
            if (c.Value is float[]) {
                float[] av = a.Value as float[], cv = c.Value as float[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = -av[i];
                }

                return c;
            }
            if (c.Value is double[]) {
                double[] av = a.Value as double[], cv = c.Value as double[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = -av[i];
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }

        /// <summary>加算</summary>
        public static NdimArray<T> operator +(NdimArray<T> a, NdimArray<T> b) {
            if (a.Shape != b.Shape) {
                throw new ArgumentException(ExceptionMessage.Shape(a.Shape, b.Shape));
            }

            NdimArray<T> c = new NdimArray<T>(a.Shape);

            if (c.Value is int[]) {
                int[] av = a.Value as int[], bv = b.Value as int[], cv = c.Value as int[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] + bv[i];
                }

                return c;
            }
            if (c.Value is long[]) {
                long[] av = a.Value as long[], bv = b.Value as long[], cv = c.Value as long[];

                for (long i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] + bv[i];
                }

                return c;
            }
            if (c.Value is float[]) {
                float[] av = a.Value as float[], bv = b.Value as float[], cv = c.Value as float[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] + bv[i];
                }

                return c;
            }
            if (c.Value is double[]) {
                double[] av = a.Value as double[], bv = b.Value as double[], cv = c.Value as double[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] + bv[i];
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }

        /// <summary>減算</summary>
        public static NdimArray<T> operator -(NdimArray<T> a, NdimArray<T> b) {
            if (a.Shape != b.Shape) {
                throw new ArgumentException(ExceptionMessage.Shape(a.Shape, b.Shape));
            }

            NdimArray<T> c = new NdimArray<T>(a.Shape);

            if (c.Value is int[]) {
                int[] av = a.Value as int[], bv = b.Value as int[], cv = c.Value as int[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] - bv[i];
                }

                return c;
            }
            if (c.Value is long[]) {
                long[] av = a.Value as long[], bv = b.Value as long[], cv = c.Value as long[];

                for (long i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] - bv[i];
                }

                return c;
            }
            if (c.Value is float[]) {
                float[] av = a.Value as float[], bv = b.Value as float[], cv = c.Value as float[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] - bv[i];
                }

                return c;
            }
            if (c.Value is double[]) {
                double[] av = a.Value as double[], bv = b.Value as double[], cv = c.Value as double[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] - bv[i];
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }

        /// <summary>乗算</summary>
        public static NdimArray<T> operator *(NdimArray<T> a, NdimArray<T> b) {
            if (a.Shape != b.Shape) {
                throw new ArgumentException(ExceptionMessage.Shape(a.Shape, b.Shape));
            }

            NdimArray<T> c = new NdimArray<T>(a.Shape);

            if (c.Value is int[]) {
                int[] av = a.Value as int[], bv = b.Value as int[], cv = c.Value as int[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] * bv[i];
                }

                return c;
            }
            if (c.Value is long[]) {
                long[] av = a.Value as long[], bv = b.Value as long[], cv = c.Value as long[];

                for (long i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] * bv[i];
                }

                return c;
            }
            if (c.Value is float[]) {
                float[] av = a.Value as float[], bv = b.Value as float[], cv = c.Value as float[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] * bv[i];
                }

                return c;
            }
            if (c.Value is double[]) {
                double[] av = a.Value as double[], bv = b.Value as double[], cv = c.Value as double[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] * bv[i];
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }

        /// <summary>除算</summary>
        public static NdimArray<T> operator /(NdimArray<T> a, NdimArray<T> b) {
            if (a.Shape != b.Shape) {
                throw new ArgumentException(ExceptionMessage.Shape(a.Shape, b.Shape));
            }

            NdimArray<T> c = new NdimArray<T>(a.Shape);

            if (c.Value is int[]) {
                int[] av = a.Value as int[], bv = b.Value as int[], cv = c.Value as int[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] / bv[i];
                }

                return c;
            }
            if (c.Value is long[]) {
                long[] av = a.Value as long[], bv = b.Value as long[], cv = c.Value as long[];

                for (long i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] / bv[i];
                }

                return c;
            }
            if (c.Value is float[]) {
                float[] av = a.Value as float[], bv = b.Value as float[], cv = c.Value as float[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] / bv[i];
                }

                return c;
            }
            if (c.Value is double[]) {
                double[] av = a.Value as double[], bv = b.Value as double[], cv = c.Value as double[];

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] / bv[i];
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }

        /// <summary>加算</summary>
        public static NdimArray<T> operator +(NdimArray<T> a, T b) {
        
            NdimArray<T> c = new NdimArray<T>(a.Shape);

            if(c.Value is int[]){
                int[] av = a.Value as int[], cv = c.Value as int[];
                int bv = (int)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] + bv;
                }

                return c;
            }
            if(c.Value is long[]){
                long[] av = a.Value as long[], cv = c.Value as long[];
                long bv = (long)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] + bv;
                }

                return c;
            }
            if(c.Value is float[]){
                float[] av = a.Value as float[], cv = c.Value as float[];
                float bv = (float)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] + bv;
                }

                return c;
            }
            if(c.Value is double[]){
                double[] av = a.Value as double[], cv = c.Value as double[];
                double bv = (double)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] + bv;
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }

        /// <summary>加算</summary>
        public static NdimArray<T> operator +(T a, NdimArray<T> b) {
            return b + a;
        }

        /// <summary>減算</summary>
        public static NdimArray<T> operator -(NdimArray<T> a, T b) {

            NdimArray<T> c = new NdimArray<T>(a.Shape);

            if(c.Value is int[]){
                int[] av = a.Value as int[], cv = c.Value as int[];
                int bv = (int)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] - bv;
                }

                return c;
            }
            if(c.Value is long[]){
                long[] av = a.Value as long[], cv = c.Value as long[];
                long bv = (long)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] - bv;
                }

                return c;
            }
            if(c.Value is float[]){
                float[] av = a.Value as float[], cv = c.Value as float[];
                float bv = (float)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] - bv;
                }

                return c;
            }
            if(c.Value is double[]){
                double[] av = a.Value as double[], cv = c.Value as double[];
                double bv = (double)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] - bv;
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }

        /// <summary>減算</summary>
        public static NdimArray<T> operator -(T a, NdimArray<T> b) {
            return -b + a;
        }

        /// <summary>乗算</summary>
        public static NdimArray<T> operator *(NdimArray<T> a, T b) {

            NdimArray<T> c = new NdimArray<T>(a.Shape);

            if(c.Value is int[]){
                int[] av = a.Value as int[], cv = c.Value as int[];
                int bv = (int)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] * bv;
                }

                return c;
            }
            if(c.Value is long[]){
                long[] av = a.Value as long[], cv = c.Value as long[];
                long bv = (long)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] * bv;
                }

                return c;
            }
            if(c.Value is float[]){
                float[] av = a.Value as float[], cv = c.Value as float[];
                float bv = (float)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] * bv;
                }

                return c;
            }
            if(c.Value is double[]){
                double[] av = a.Value as double[], cv = c.Value as double[];
                double bv = (double)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] * bv;
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }

        /// <summary>乗算</summary>
        public static NdimArray<T> operator *(T a, NdimArray<T> b) {
            return b * a;
        }

        /// <summary>除算</summary>
        public static NdimArray<T> operator /(NdimArray<T> a, T b) {

            NdimArray<T> c = new NdimArray<T>(a.Shape);

            if(c.Value is int[]){
                int[] av = a.Value as int[], cv = c.Value as int[];
                int bv = (int)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] / bv;
                }

                return c;
            }
            if(c.Value is long[]){
                long[] av = a.Value as long[], cv = c.Value as long[];
                long bv = (long)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] / bv;
                }

                return c;
            }
            if(c.Value is float[]){
                float[] av = a.Value as float[], cv = c.Value as float[];
                float bv = (float)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] / bv;
                }

                return c;
            }
            if(c.Value is double[]){
                double[] av = a.Value as double[], cv = c.Value as double[];
                double bv = (double)(object)b;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av[i] / bv;
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }

        /// <summary>除算</summary>
        public static NdimArray<T> operator /(T a, NdimArray<T> b) {

            NdimArray<T> c = new NdimArray<T>(b.Shape);

            if(c.Value is int[]){
                int[] bv = b.Value as int[], cv = c.Value as int[];
                int av = (int)(object)a;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av / bv[i];
                }

                return c;
            }
            if(c.Value is long[]){
                long[] bv = b.Value as long[], cv = c.Value as long[];
                long av = (long)(object)a;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av / bv[i];
                }

                return c;
            }
            if(c.Value is float[]){
                float[] bv = b.Value as float[], cv = c.Value as float[];
                float av = (float)(object)a;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av / bv[i];
                }

                return c;
            }
            if(c.Value is double[]){
                double[] bv = b.Value as double[], cv = c.Value as double[];
                double av = (double)(object)a;

                for (int i = 0; i < cv.Length; i++) {
                    cv[i] = av / bv[i];
                }

                return c;
            }

            throw new ArgumentException(ExceptionMessage.InvalidType());
        }
    }
}
