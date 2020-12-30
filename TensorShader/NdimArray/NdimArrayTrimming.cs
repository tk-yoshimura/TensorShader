using System;

namespace TensorShader {

    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> {

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming(NdimArray<T> arr, params (int, int)[] trims) {
            if (arr.Ndim != trims.Length) {
                throw new ArgumentException(ExceptionMessage.Argument(nameof(trims), arr.Ndim, trims.Length));
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

                int length = arr.Shape.Stride(axis + 1), new_length = new_arr.Shape.Stride(axis + 1);
                int n = arr.Length / length, sft = ta * arr.Shape.Stride(axis);

                for (int i = 0; i < n; i++) {
                    Array.Copy(arr.Value, i * length + sft, new_arr.Value, i * new_length, new_length);
                }

                arr = new_arr;
            }

            return arr;
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming1D(NdimArray<T> arr, int trim_left, int trim_right) {
            if (arr.Type != ShapeType.Map || arr.Ndim != 3) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            return Trimming(arr, (0, 0), (trim_left, trim_right), (0, 0));
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming1D(NdimArray<T> arr, int trim) {
            return Trimming1D(arr, trim, trim);
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming2D(NdimArray<T> arr, int trim_left, int trim_right, int trim_top, int trim_bottom) {
            if (arr.Type != ShapeType.Map || arr.Ndim != 4) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", 4), ("Type", ShapeType.Map)));
            }

            return Trimming(arr, (0, 0), (trim_left, trim_right), (trim_top, trim_bottom), (0, 0));
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming2D(NdimArray<T> arr, int trim) {
            return Trimming2D(arr, trim, trim, trim, trim);
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming3D(NdimArray<T> arr, int trim_left, int trim_right, int trim_top, int trim_bottom, int trim_front, int trim_rear) {
            if (arr.Type != ShapeType.Map || arr.Ndim != 5) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", 5), ("Type", ShapeType.Map)));
            }

            return Trimming(arr, (0, 0), (trim_left, trim_right), (trim_top, trim_bottom), (trim_front, trim_rear), (0, 0));
        }

        /// <summary>トリミング</summary>
        public static NdimArray<T> Trimming3D(NdimArray<T> arr, int trim) {
            return Trimming3D(arr, trim, trim, trim, trim, trim, trim);
        }
    }
}
