using System;

namespace TensorShader {

    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> {

        /// <summary>ゼロパディング</summary>
        public static NdimArray<T> ZeroPadding(NdimArray<T> arr, params (int, int)[] pads) {
            if (arr.Ndim != pads.Length) {
                throw new ArgumentException(ExceptionMessage.Argument(nameof(pads), arr.Ndim, pads.Length));
            }

            foreach ((int pa, int pb) in pads) {
                if (pa < 0 || pb < 0) {
                    throw new ArgumentException(nameof(pads));
                }
            }

            for (int axis = 0; axis < arr.Ndim; axis++) {
                (int pa, int pb) = pads[axis];

                if (pa <= 0 && pb <= 0) {
                    continue;
                }

                int[] s = arr.Shape;
                s[axis] += pa + pb;
                NdimArray<T> new_arr = new Shape(arr.Type, s);

                long length = arr.Shape.Stride(axis + 1), new_length = new_arr.Shape.Stride(axis + 1);
                long n = arr.Length / length, sft = pa * new_arr.Shape.Stride(axis);

                for (long i = 0; i < n; i++) {
                    Array.Copy(arr.Value, i * length, new_arr.Value, i * new_length + sft, length);
                }

                arr = new_arr;
            }

            return arr;
        }

        /// <summary>エッジパディング</summary>
        public static NdimArray<T> EdgePadding(NdimArray<T> arr, params (int, int)[] pads) {
            if (arr.Ndim != pads.Length) {
                throw new ArgumentException(ExceptionMessage.Argument(nameof(pads), arr.Ndim, pads.Length));
            }

            foreach ((int pa, int pb) in pads) {
                if (pa < 0 || pb < 0) {
                    throw new ArgumentException(nameof(pads));
                }
            }

            for (int axis = 0; axis < arr.Ndim; axis++) {
                (int pa, int pb) = pads[axis];

                if (pa <= 0 && pb <= 0) {
                    continue;
                }

                int[] s = arr.Shape;
                int l = s[axis];

                s[axis] += pa + pb;
                NdimArray<T> new_arr = new Shape(arr.Type, s);

                long stride = arr.Shape.Stride(axis), new_stride = new_arr.Shape.Stride(axis);
                long length = arr.Shape.Stride(axis + 1), new_length = new_arr.Shape.Stride(axis + 1);
                long n = arr.Length / length, sft = pa * new_stride;

                for (long i = 0; i < n; i++) {
                    Array.Copy(arr.Value, i * length, new_arr.Value, i * new_length + sft, length);
                }

                for (long i = 0; i < n; i++) {
                    for (int a = 0; a < pa; a++) {
                        Array.Copy(arr.Value, i * length, new_arr.Value, i * new_length + a * new_stride, stride);
                    }
                    for (int b = 0; b < pb; b++) {
                        Array.Copy(arr.Value, i * length + (l - 1) * stride, new_arr.Value, i * new_length + (l + pa + b) * new_stride, stride);
                    }
                }

                arr = new_arr;
            }

            return arr;
        }

        /// <summary>ゼロパディング</summary>
        public static NdimArray<T> ZeroPadding1D(NdimArray<T> arr, int pad_left, int pad_right) {
            if (arr.Type != ShapeType.Map || arr.Ndim != 3) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            return ZeroPadding(arr, (0, 0), (pad_left, pad_right), (0, 0));
        }

        /// <summary>ゼロパディング</summary>
        public static NdimArray<T> ZeroPadding1D(NdimArray<T> arr, int pad) {
            return ZeroPadding1D(arr, pad, pad);
        }

        /// <summary>ゼロパディング</summary>
        public static NdimArray<T> ZeroPadding2D(NdimArray<T> arr, int pad_left, int pad_right, int pad_top, int pad_bottom) {
            if (arr.Type != ShapeType.Map || arr.Ndim != 4) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", 4), ("Type", ShapeType.Map)));
            }

            return ZeroPadding(arr, (0, 0), (pad_left, pad_right), (pad_top, pad_bottom), (0, 0));
        }

        /// <summary>ゼロパディング</summary>
        public static NdimArray<T> ZeroPadding2D(NdimArray<T> arr, int pad) {
            return ZeroPadding2D(arr, pad, pad, pad, pad);
        }

        /// <summary>ゼロパディング</summary>
        public static NdimArray<T> ZeroPadding3D(NdimArray<T> arr, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_rear) {
            if (arr.Type != ShapeType.Map || arr.Ndim != 5) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", 5), ("Type", ShapeType.Map)));
            }

            return ZeroPadding(arr, (0, 0), (pad_left, pad_right), (pad_top, pad_bottom), (pad_front, pad_rear), (0, 0));
        }

        /// <summary>ゼロパディング</summary>
        public static NdimArray<T> ZeroPadding3D(NdimArray<T> arr, int pad) {
            return ZeroPadding3D(arr, pad, pad, pad, pad, pad, pad);
        }

        /// <summary>エッジパディング</summary>
        public static NdimArray<T> EdgePadding1D(NdimArray<T> arr, int pad_left, int pad_right) {
            if (arr.Type != ShapeType.Map || arr.Ndim != 3) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", 3), ("Type", ShapeType.Map)));
            }

            return EdgePadding(arr, (0, 0), (pad_left, pad_right), (0, 0));
        }

        /// <summary>エッジパディング</summary>
        public static NdimArray<T> EdgePadding1D(NdimArray<T> arr, int pad) {
            return EdgePadding1D(arr, pad, pad);
        }

        /// <summary>エッジパディング</summary>
        public static NdimArray<T> EdgePadding2D(NdimArray<T> arr, int pad_left, int pad_right, int pad_top, int pad_bottom) {
            if (arr.Type != ShapeType.Map || arr.Ndim != 4) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", 4), ("Type", ShapeType.Map)));
            }

            return EdgePadding(arr, (0, 0), (pad_left, pad_right), (pad_top, pad_bottom), (0, 0));
        }

        /// <summary>エッジパディング</summary>
        public static NdimArray<T> EdgePadding2D(NdimArray<T> arr, int pad) {
            return EdgePadding2D(arr, pad, pad, pad, pad);
        }

        /// <summary>エッジパディング</summary>
        public static NdimArray<T> EdgePadding3D(NdimArray<T> arr, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_rear) {
            if (arr.Type != ShapeType.Map || arr.Ndim != 5) { 
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim", 5), ("Type", ShapeType.Map)));
            }

            return EdgePadding(arr, (0, 0), (pad_left, pad_right), (pad_top, pad_bottom), (pad_front, pad_rear), (0, 0));
        }

        /// <summary>エッジパディング</summary>
        public static NdimArray<T> EdgePadding3D(NdimArray<T> arr, int pad) {
            return EdgePadding3D(arr, pad, pad, pad, pad, pad, pad);
        }
    }
}
