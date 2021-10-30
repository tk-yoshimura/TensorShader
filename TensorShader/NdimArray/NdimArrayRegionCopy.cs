using System;
using System.Linq;

namespace TensorShader {

    /// <summary>多次元配列</summary>
    public partial class NdimArray<T> {

        /// <summary>領域上書き</summary>
        public static void RegionCopy(NdimArray<T> src_arr, NdimArray<T> dst_arr, params int[] ps) {
            if (src_arr.Ndim != ps.Length) {
                throw new ArgumentException(ExceptionMessage.Argument($"{nameof(ps)}.Length", src_arr.Ndim, ps.Length));
            }

            if (src_arr.Ndim != dst_arr.Ndim || src_arr.Type != dst_arr.Type) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(dst_arr.Shape, ("Ndim", src_arr.Ndim), ("Type", src_arr.Type)));
            }

            int ndim = src_arr.Ndim;

            for (int i = 0; i < ndim; i++) {
                int p = ps[i];

                if (p < 0 || p + src_arr.Shape[i] > dst_arr.Shape[i]) {
                    throw new ArgumentOutOfRangeException(nameof(ps));
                }
            }

            int stride_axis = 0;
            for (int i = 0; i < ndim; i++) {
                if (src_arr.Shape[i] != dst_arr.Shape[i]) {
                    break;
                }
                stride_axis++;
            }

            if (stride_axis >= ndim) {
                Array.Copy(src_arr.Value, 0, dst_arr.Value, 0, src_arr.Length);
                return;
            }

            int src_stride = src_arr.Shape.Stride(stride_axis + 1), dst_stride = dst_arr.Shape.Stride(stride_axis + 1);
            int l0 = (stride_axis + 1 < ndim) ? src_arr.Shape[stride_axis + 1] : 1;
            int l1 = (stride_axis + 2 < ndim) ? src_arr.Shape[stride_axis + 2] : 1;
            int n = src_arr.Length / (l0 * l1 * src_stride);

            long src_index = 0;
            int[] ss = new int[ndim + 3];
            int[] ms = (int[])ps.Clone();

            for (int i = 0; i < n; i++) {
                long dst_org = dst_arr.Index(ms);

                for (int j = 0; j < l1; j++) {
                    long dst_index = dst_org;

                    for (int k = 0; k < l0; k++) {
                        Array.Copy(src_arr.Value, src_index, dst_arr.Value, dst_index, src_stride);
                        src_index += src_stride;
                        dst_index += dst_stride;
                    }

                    dst_org += dst_stride * ((stride_axis + 1 < ndim) ? dst_arr.Shape[stride_axis + 1] : 1);
                }

                ss[stride_axis + 3]++;
                for (int j = stride_axis + 3; j < src_arr.Ndim; j++) {
                    if (ss[j] < src_arr.Shape[j]) {
                        break;
                    }
                    ss[j] = 0;
                    ss[j + 1]++;
                }

                ms = ms.Select((_, idx) => ps[idx] + ss[idx]).ToArray();
            }
        }

        /// <summary>領域上書き</summary>
        public static void RegionCopyND(NdimArray<T> src_arr, NdimArray<T> dst_arr, params int[] ps) {
            if (src_arr.Type != ShapeType.Map || src_arr.Ndim != ps.Length + 2) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(src_arr.Shape, ("Ndim", ps.Length + 2), ("Type", ShapeType.Map)));
            }
            if (dst_arr.Type != ShapeType.Map || dst_arr.Ndim != ps.Length + 2) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(dst_arr.Shape, ("Ndim", ps.Length + 2), ("Type", ShapeType.Map)));
            }
            if (src_arr.Channels != dst_arr.Channels || src_arr.Batch != dst_arr.Batch) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(dst_arr.Shape, ("Channels", src_arr.Channels), ("Batch", src_arr.Batch)));
            }

            int[] new_ps = (new int[] { 0 }).Concat(ps).Concat(new int[] { 0 }).ToArray();

            RegionCopy(src_arr, dst_arr, new_ps);
        }

        /// <summary>領域上書き</summary>
        public static void RegionCopy1D(NdimArray<T> src_arr, NdimArray<T> dst_arr, int x) {
            RegionCopyND(src_arr, dst_arr, x);
        }

        /// <summary>領域上書き</summary>
        public static void RegionCopy2D(NdimArray<T> src_arr, NdimArray<T> dst_arr, int x, int y) {
            RegionCopyND(src_arr, dst_arr, x, y);
        }

        /// <summary>領域上書き</summary>
        public static void RegionCopy3D(NdimArray<T> src_arr, NdimArray<T> dst_arr, int x, int y, int z) {
            RegionCopyND(src_arr, dst_arr, x, y, z);
        }

        /// <summary>領域上書き</summary>
        public static void RegionCopy(NdimArray<T> src_arr, NdimArray<T> dst_arr, params (int src_offset, int dst_offset, int count)[] region) {
            NdimArray<T> src_sliced_arr = Slice(src_arr, region.Select((r) => (r.src_offset, r.count)).ToArray());
            RegionCopy(src_sliced_arr, dst_arr, region.Select((r) => r.dst_offset).ToArray());
        }

        /// <summary>領域上書き</summary>
        public static void RegionCopyND(NdimArray<T> src_arr, NdimArray<T> dst_arr, params (int src_offset, int dst_offset, int count)[] region) {
            if (src_arr.Type != ShapeType.Map || src_arr.Ndim != region.Length + 2) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(src_arr.Shape, ("Ndim", region.Length + 2), ("Type", ShapeType.Map)));
            }

            (int, int, int)[] new_region =
                (new (int, int, int)[] { (0, 0, src_arr.Channels) })
                .Concat(region)
                .Concat(new (int, int, int)[] { (0, 0, src_arr.Batch) }).ToArray();


            RegionCopy(src_arr, dst_arr, new_region);
        }

        /// <summary>領域上書き</summary>
        public static void RegionCopy1D(NdimArray<T> src_arr, NdimArray<T> dst_arr, int src_xoffset, int dst_xoffset, int xcount) {
            RegionCopyND(src_arr, dst_arr, (src_xoffset, dst_xoffset, xcount));
        }

        /// <summary>領域上書き</summary>
        public static void RegionCopy2D(NdimArray<T> src_arr, NdimArray<T> dst_arr, int src_xoffset, int dst_xoffset, int xcount, int src_yoffset, int dst_yoffset, int ycount) {
            RegionCopyND(src_arr, dst_arr, (src_xoffset, dst_xoffset, xcount), (src_yoffset, dst_yoffset, ycount));
        }

        /// <summary>領域上書き</summary>
        public static void RegionCopy3D(NdimArray<T> src_arr, NdimArray<T> dst_arr, int src_xoffset, int dst_xoffset, int xcount, int src_yoffset, int dst_yoffset, int ycount, int src_zoffset, int dst_zoffset, int zcount) {
            RegionCopyND(src_arr, dst_arr, (src_xoffset, dst_xoffset, xcount), (src_yoffset, dst_yoffset, ycount), (src_zoffset, dst_zoffset, zcount));
        }
    }
}
