using System;
using System.Linq;

namespace TensorShader {

    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> {

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming(NdimArray<T> arr, params (int, int)[] trims) {
            if (arr.Ndim != trims.Length) {
                throw new ArgumentException(ExceptionMessage.Argument($"{nameof(trims)}.Length", arr.Ndim, trims.Length));
            }

            for(int i = 0; i < arr.Ndim; i++) {
                (int ta, int tb) = trims[i];

                if (ta < 0 || tb < 0 || (ta + tb) >= arr.Shape[i]) {
                    throw new ArgumentException(nameof(trims));
                }
            }

            for (int axis = arr.Ndim - 1; axis >= 0; axis--) {
                (int ta, int tb) = trims[axis];

                if (ta <= 0 && tb <= 0) {
                    continue;
                }

                int[] s = arr.Shape;
                s[axis] -= ta + tb;
                NdimArray<T> new_arr = new Shape(arr.Type, s);

                long length = arr.Shape.Stride(axis + 1), new_length = new_arr.Shape.Stride(axis + 1);
                long n = arr.Length / length, sft = ta * arr.Shape.Stride(axis);

                for (long i = 0; i < n; i++) {
                    Array.Copy(arr.Value, i * length + sft, new_arr.Value, i * new_length, new_length);
                }

                arr = new_arr;
            }

            return arr;
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> TrimmingND(NdimArray<T> arr, params (int, int)[] trims) {
            if (arr.Type != ShapeType.Map || arr.Ndim != trims.Length + 2) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", trims.Length + 2), ("Type", ShapeType.Map)));
            }

            (int, int)[] new_trims = (new (int, int)[] { (0, 0) }).Concat(trims).Concat(new (int, int)[] { (0, 0) }).ToArray();

            return Trimming(arr, new_trims);
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> TrimmingND(NdimArray<T> arr, int trim) {
            if (arr.Type != ShapeType.Map || arr.Ndim >= 3) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim>", 3), ("Type", ShapeType.Map)));
            }

            (int, int)[] new_trims = (new (int, int)[] { (0, 0) }).Concat(Enumerable.Repeat((trim, trim), arr.Ndim - 2)).Concat(new (int, int)[] { (0, 0) }).ToArray();

            return Trimming(arr, new_trims);
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming1D(NdimArray<T> arr, int trim_left, int trim_right) {
            return TrimmingND(arr, (trim_left, trim_right));
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming1D(NdimArray<T> arr, int trim) {
            return TrimmingND(arr, trim);
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming2D(NdimArray<T> arr, int trim_left, int trim_right, int trim_top, int trim_bottom) {
            return TrimmingND(arr, (trim_left, trim_right), (trim_top, trim_bottom));
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming2D(NdimArray<T> arr, int trim) {
            return TrimmingND(arr, trim);
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming3D(NdimArray<T> arr, int trim_left, int trim_right, int trim_top, int trim_bottom, int trim_front, int trim_rear) {
            return TrimmingND(arr, (trim_left, trim_right), (trim_top, trim_bottom), (trim_front, trim_rear));
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming3D(NdimArray<T> arr, int trim) {
            return TrimmingND(arr, trim);
        }

        /// <summary>切り取り</summary>
        public static NdimArray<T> Slice(NdimArray<T> arr, params (int offset, int count)[] range) {
            if (arr.Ndim != range.Length) {
                throw new ArgumentException(ExceptionMessage.Argument($"{nameof(range)}.Length", arr.Ndim, range.Length));
            }

            return Trimming(arr, range.Select((r, i) => (r.offset, arr.Shape[i] - r.offset - r.count)).ToArray());
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> SliceND(NdimArray<T> arr, params (int offset, int count)[] range) {
            if (arr.Type != ShapeType.Map || arr.Ndim != range.Length + 2) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", range.Length + 2), ("Type", ShapeType.Map)));
            }

            (int, int)[] new_range = (new (int, int)[] { (0, arr.Channels) }).Concat(range).Concat(new (int, int)[] { (0, arr.Batch) }).ToArray();

            return Slice(arr, new_range);
        }

        /// <summary>切り取り</summary>
        public static NdimArray<T> Slice1D(NdimArray<T> arr, int x, int width) {
            return SliceND(arr, (x, width));
        }

        /// <summary>切り取り</summary>
        public static NdimArray<T> Slice2D(NdimArray<T> arr, int x, int width, int y, int height) {
            return SliceND(arr, (x, width), (y, height));
        }

        /// <summary>切り取り</summary>
        public static NdimArray<T> Slice3D(NdimArray<T> arr, int x, int width, int y, int height, int z, int depth) {
            return SliceND(arr, (x, width), (y, height), (z, depth));
        }
    }
}
