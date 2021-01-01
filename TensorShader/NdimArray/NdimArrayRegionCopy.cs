using System;
using System.Linq;

namespace TensorShader {

    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> {

        /// <summary>領域上書き</summary>
        public static void RegionCopy(NdimArray<T> src_arr, NdimArray<T> dst_arr, params int[] ps) {
            if (src_arr.Ndim != ps.Length) {
                throw new ArgumentException(ExceptionMessage.Argument(nameof(ps), src_arr.Ndim, ps.Length));
            }

            if (src_arr.Ndim != dst_arr.Ndim) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(dst_arr.Shape, ("Ndim", src_arr.Ndim)));
            }

            for(int i = 0; i < src_arr.Ndim; i++) {
                int p = ps[i];

                if (p < 0 || p + src_arr.Shape[i] > dst_arr.Shape[i]) {
                    throw new ArgumentOutOfRangeException(nameof(ps));
                }
            }

            int stride_axis = 0;
            for(int i = 0; i < src_arr.Ndim; i++) {
                if (src_arr.Shape[i] != dst_arr.Shape[i]) {
                    break;
                }
                stride_axis++;
            }

            if (stride_axis >= src_arr.Ndim) {
                Array.Copy(src_arr.Value, 0, dst_arr.Value, 0, src_arr.Length);
                return;
            }

            int src_stride = src_arr.Shape.Stride(stride_axis + 1), dst_stride = dst_arr.Shape.Stride(stride_axis + 1);
            int l = src_arr.Shape[stride_axis + 1], n = src_arr.Length / (l * src_stride);

            long src_index = 0;
            int[] ss = new int[src_arr.Ndim + 1];
            int[] ms = (int[])ps.Clone();
            
            for (int i = 0; i < n; i++) {
                long dst_index = dst_arr.Index(ms);

                for (int j = 0; j < l; j++) {
                    Array.Copy(src_arr.Value, src_index, dst_arr.Value, dst_index, src_stride);
                    src_index += src_stride;
                    dst_index += dst_stride;
                }

                ss[stride_axis + 2]++;
                for (int j = stride_axis + 2; j < src_arr.Ndim; j++) {
                    if (ss[j] < src_arr.Shape[j]) {
                        break;
                    }
                    ss[j] = 0;
                    ss[j + 1]++;
                }

                ms = ms.Select((_, idx) => ps[idx] + ss[idx]).ToArray();
            }
        }
    }
}
