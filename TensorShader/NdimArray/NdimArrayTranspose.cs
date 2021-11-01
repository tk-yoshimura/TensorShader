using System;
using System.Linq;

namespace TensorShader {

    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> {

        /// <summary>転置</summary>
        public static NdimArray<T> Transpose(NdimArray<T> arr, (int, int) axis) {
            return Transpose(arr, axis.Item1, axis.Item2);
        }

        /// <summary>転置</summary>
        public static NdimArray<T> Transpose(NdimArray<T> arr, (int, int, int) axis) {
            return Transpose(arr, axis.Item1, axis.Item2, axis.Item3);
        }

        /// <summary>転置</summary>
        public static NdimArray<T> Transpose(NdimArray<T> arr, (int, int, int, int) axis) {
            return Transpose(arr, axis.Item1, axis.Item2, axis.Item3, axis.Item4);
        }

        /// <summary>転置</summary>
        public static NdimArray<T> Transpose(NdimArray<T> arr, (int, int, int, int, int) axis) {
            return Transpose(arr, axis.Item1, axis.Item2, axis.Item3, axis.Item4, axis.Item5);
        }

        /// <summary>転置</summary>
        public static NdimArray<T> Transpose(NdimArray<T> arr, (int, int, int, int, int, int) axis) {
            return Transpose(arr, axis.Item1, axis.Item2, axis.Item3, axis.Item4, axis.Item5, axis.Item6);
        }

        /// <summary>転置</summary>
        public static NdimArray<T> Transpose(NdimArray<T> arr, (int, int, int, int, int, int, int) axis) {
            return Transpose(arr, axis.Item1, axis.Item2, axis.Item3, axis.Item4, axis.Item5, axis.Item6, axis.Item7);
        }

        /// <summary>転置</summary>
        public static NdimArray<T> Transpose(NdimArray<T> arr, (int, int, int, int, int, int, int, int) axis) {
            return Transpose(arr, axis.Item1, axis.Item2, axis.Item3, axis.Item4, axis.Item5, axis.Item6, axis.Item7, axis.Item8);
        }

        private static NdimArray<T> Transpose(NdimArray<T> arr, params int[] axis) {
            if (arr.Ndim != axis.Length) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(arr.Shape, ("Ndim==", axis.Length)));
            }

            int[] axis_sorted = axis.OrderBy((i) => i).ToArray();
            for (int i = 0; i < arr.Ndim; i++) {
                if (axis_sorted[i] != i) {
                    throw new ArgumentException(null, nameof(axis));
                }
            }

            static int[] permutate(int[] axis) {
                int[] new_axis = new int[axis.Length];

                for (int i = 0; i < axis.Length; i++) {
                    new_axis[axis[i]] = i;
                }

                return new_axis;
            }

            static (int block, int n, int m, int s) compute_stride(int[] axis_length, int axis_index1, int axis_index2) {
                int block = 1, n = 1, m = 1, s = 1;
                for (int i = 0; i < axis_index1; i++) {
                    block *= axis_length[i];
                }
                for (int i = axis_index1; i < axis_index2; i++) {
                    n *= axis_length[i];
                }
                m = axis_length[axis_index2];
                for (int i = axis_index2 + 1; i < axis_length.Length; i++) {
                    s *= axis_length[i];
                }

                return (block, n, m, s);
            }

            static (int[] axis, int[] axis_length) update_axis(int[] axis, int[] axis_length, int axis_index1, int axis_index2) {
                int[] new_axis = (int[])axis.Clone();
                int[] new_axis_length = (int[])axis_length.Clone();

                new_axis[axis_index1] = axis[axis_index2];
                new_axis_length[axis_index1] = axis_length[axis_index2];

                for (int i = axis_index1 + 1; i <= axis_index2; i++) {
                    new_axis[i] = axis[i - 1];
                    new_axis_length[i] = axis_length[i - 1];
                }

                return (new_axis, new_axis_length);
            }

            T[] vs = arr.Value;
            int[] axis_length = arr.Shape;
            axis = permutate(axis);

            for (int axis_index1 = 0; axis_index1 < arr.Ndim; axis_index1++) {
                if (axis[axis_index1] != axis_index1) {
                    for (int axis_index2 = axis_index1 + 1; axis_index2 < arr.Ndim; axis_index2++) {
                        if (axis[axis_index2] == axis_index1) {

                            (int block, int n, int m, int s) = compute_stride(axis_length, axis_index1, axis_index2);
                            vs = Transpose(vs, block, n, m, s);
                            (axis, axis_length) = update_axis(axis, axis_length, axis_index1, axis_index2);

                        }
                    }
                }
            }

            return new NdimArray<T>(new Shape(arr.Shape.Type, axis_length), vs, clone_value: false);
        }

        private static T[] Transpose(T[] vs, int block, int n, int m, int s) {
            int length = checked(block * n * m * s);

            if (vs.Length != length) {
                throw new ArgumentException($"{nameof(vs)}.Length");
            }

            T[] us = new T[length];

            if (block > 1) {
                for (int k = 0, src_idx = 0; k < s; k++) {
                    for (int j = 0; j < m; j++) {
                        for (int i = 0; i < n; i++, src_idx += block) {
                            int dst_idx = block * (j + m * (i + n * k));
                            Array.Copy(vs, src_idx, us, dst_idx, block);
                        }
                    }
                }
            }
            else {
                for (int k = 0, src_idx = 0; k < s; k++) {
                    for (int j = 0; j < m; j++) {
                        for (int i = 0; i < n; i++, src_idx++) {
                            int dst_idx = j + m * (i + n * k);
                            us[dst_idx] = vs[src_idx];
                        }
                    }
                }
            }

            return us;
        }
    }
}
