using System;
using System.Runtime.InteropServices;

namespace TensorShader {

    /// <summary>多次元配列1次元化拡張</summary>
    public static class ArrayFlattenExtensions {

        private static int Shape<T>(this T[] v) where T : struct, IComparable {
            return v.GetLength(0);
        }

        private static (int, int) Shape<T>(this T[,] v) where T : struct, IComparable {
            return (v.GetLength(1), v.GetLength(0));
        }

        private static (int, int, int) Shape<T>(this T[,,] v) where T : struct, IComparable {
            return (v.GetLength(2), v.GetLength(1), v.GetLength(0));
        }

        private static (int, int, int, int) Shape<T>(this T[,,,] v) where T : struct, IComparable {
            return (v.GetLength(3), v.GetLength(2), v.GetLength(1), v.GetLength(0));
        }

        private static (int, int, int, int, int) Shape<T>(this T[,,,,] v) where T : struct, IComparable {
            return (v.GetLength(4), v.GetLength(3), v.GetLength(2), v.GetLength(1), v.GetLength(0));
        }

        private static void BlockCopyTo<T>(this T[] vs, T[] us, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(vs, 0, us, index * Marshal.SizeOf<T>(), vs.Length * Marshal.SizeOf<T>());
        }

        private static void BlockCopyTo<T>(this T[,] vs, T[] us, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(vs, 0, us, index * Marshal.SizeOf<T>(), vs.Length * Marshal.SizeOf<T>());
        }

        private static void BlockCopyTo<T>(this T[,,] vs, T[] us, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(vs, 0, us, index * Marshal.SizeOf<T>(), vs.Length * Marshal.SizeOf<T>());
        }

        private static void BlockCopyTo<T>(this T[,,,] vs, T[] us, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(vs, 0, us, index * Marshal.SizeOf<T>(), vs.Length * Marshal.SizeOf<T>());
        }

        private static void BlockCopyTo<T>(this T[,,,,] vs, T[] us, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(vs, 0, us, index * Marshal.SizeOf<T>(), vs.Length * Marshal.SizeOf<T>());
        }

        private static void BlockCopyTo<T>(this T[,,,,,] vs, T[] us, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(vs, 0, us, index * Marshal.SizeOf<T>(), vs.Length * Marshal.SizeOf<T>());
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[][] vs) where T : struct, IComparable {
            int batch = vs.Length, size = vs[0].Length;
            int shape = vs[0].Shape();

            for (int i = 1; i < vs.Length; i++) {
                if (vs[i].Shape() != shape) {
                    throw new ArgumentException(ExceptionMessage.ShapeWithIndex(i, $"{vs[i].Shape()}", $"{shape}"));
                }
            }

            T[] us = new T[batch * size];
            for (int i = 0; i < vs.Length; i++) {
                vs[i].BlockCopyTo(us, i * size);
            }

            return us;
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[,,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[][,] vs) where T : struct, IComparable {
            int batch = vs.Length, size = vs[0].Length;
            (int, int) shape = vs[0].Shape();

            for (int i = 1; i < vs.Length; i++) {
                if (vs[i].Shape() != shape) {
                    throw new ArgumentException(ExceptionMessage.ShapeWithIndex(i, $"{vs[i].Shape()}", $"{shape}"));
                }
            }

            T[] us = new T[batch * size];
            for (int i = 0; i < vs.Length; i++) {
                vs[i].BlockCopyTo(us, i * size);
            }

            return us;
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[,,,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[][,,] vs) where T : struct, IComparable {
            int batch = vs.Length, size = vs[0].Length;
            (int, int, int) shape = vs[0].Shape();

            for (int i = 1; i < vs.Length; i++) {
                if (vs[i].Shape() != shape) {
                    throw new ArgumentException(ExceptionMessage.ShapeWithIndex(i, $"{vs[i].Shape()}", $"{shape}"));
                }
            }

            T[] us = new T[batch * size];
            for (int i = 0; i < vs.Length; i++) {
                vs[i].BlockCopyTo(us, i * size);
            }

            return us;
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[,,,,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[][,,,] vs) where T : struct, IComparable {
            int batch = vs.Length, size = vs[0].Length;
            (int, int, int, int) shape = vs[0].Shape();

            for (int i = 1; i < vs.Length; i++) {
                if (vs[i].Shape() != shape) {
                    throw new ArgumentException(ExceptionMessage.ShapeWithIndex(i, $"{vs[i].Shape()}", $"{shape}"));
                }
            }

            T[] us = new T[batch * size];
            for (int i = 0; i < vs.Length; i++) {
                vs[i].BlockCopyTo(us, i * size);
            }

            return us;
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[,,,,,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>1次元化</summary>
        public static T[] Flatten<T>(this T[][,,,,] vs) where T : struct, IComparable {
            int batch = vs.Length, size = vs[0].Length;
            (int, int, int, int, int) shape = vs[0].Shape();

            for (int i = 1; i < vs.Length; i++) {
                if (vs[i].Shape() != shape) {
                    throw new ArgumentException(ExceptionMessage.ShapeWithIndex(i, $"{vs[i].Shape()}", $"{shape}"));
                }
            }

            T[] us = new T[batch * size];
            for (int i = 0; i < vs.Length; i++) {
                vs[i].BlockCopyTo(us, i * size);
            }

            return us;
        }
    }
}
