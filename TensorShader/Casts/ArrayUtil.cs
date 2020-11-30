using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace TensorShader {

    /// <summary>多次元配列平坦配列化拡張</summary>
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

        private static void BlockCopyTo<T>(this T[] us, T[,] vs, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(us, index * Marshal.SizeOf<T>(), vs, 0, vs.Length * Marshal.SizeOf<T>());
        }

        private static void BlockCopyTo<T>(this T[] us, T[,,] vs, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(us, index * Marshal.SizeOf<T>(), vs, 0, vs.Length * Marshal.SizeOf<T>());
        }

        private static void BlockCopyTo<T>(this T[] us, T[,,,] vs, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(us, index * Marshal.SizeOf<T>(), vs, 0, vs.Length * Marshal.SizeOf<T>());
        }

        private static void BlockCopyTo<T>(this T[] us, T[,,,,] vs, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(us, index * Marshal.SizeOf<T>(), vs, 0, vs.Length * Marshal.SizeOf<T>());
        }

        private static void BlockCopyTo<T>(this T[] us, T[,,,,,] vs, int index) where T : struct, IComparable {
            if (index + vs.Length > us.Length) {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            Buffer.BlockCopy(us, index * Marshal.SizeOf<T>(), vs, 0, vs.Length * Marshal.SizeOf<T>());
        }

        /// <summary>平坦配列化</summary>
        public static T[] Flatten<T>(this T[,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>平坦配列化</summary>
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

        /// <summary>平坦配列化</summary>
        public static T[] Flatten<T>(this T[,,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>平坦配列化</summary>
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

        /// <summary>平坦配列化</summary>
        public static T[] Flatten<T>(this T[,,,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>平坦配列化</summary>
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

        /// <summary>平坦配列化</summary>
        public static T[] Flatten<T>(this T[,,,,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>平坦配列化</summary>
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

        /// <summary>平坦配列化</summary>
        public static T[] Flatten<T>(this T[,,,,,] vs) where T : struct, IComparable {
            T[] us = new T[vs.Length];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>平坦配列化</summary>
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

        /// <summary>2次元配列化</summary>
        public static T[,] To2DArray<T>(this T[] vs, (int, int) length) where T : struct, IComparable {
            if (vs.Length != length.Item1 * length.Item2) {
                throw new ArgumentException(nameof(length));
            }

            T[,] us = new T[length.Item2, length.Item1];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>1次元配列配列化</summary>
        public static T[][] To1DArrays<T>(this T[] vs, int length, int batch) where T : struct, IComparable {
            if (vs.Length != length * batch) {
                throw new ArgumentException(nameof(length));
            }

            T[][] us = (new T[batch][]).Select((_) => new T[length]).ToArray();
            
            for (int i = 0, size = length; i < us.Length; i++) {
                vs.BlockCopyTo(us[i], i * size);
            }

            return us;
        }

        /// <summary>3次元配列化</summary>
        public static T[,,] To3DArray<T>(this T[] vs, (int, int, int) length) where T : struct, IComparable {
            if (vs.Length != length.Item1 * length.Item2 * length.Item3) {
                throw new ArgumentException(nameof(length));
            }

            T[,,] us = new T[length.Item3, length.Item2, length.Item1];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>2次元配列配列化</summary>
        public static T[][,] To2DArrays<T>(this T[] vs, (int, int) length, int batch) where T : struct, IComparable {
            if (vs.Length != length.Item1 * length.Item2 * batch) {
                throw new ArgumentException(nameof(length));
            }

            T[][,] us = (new T[batch][,]).Select((_) => new T[length.Item2, length.Item1]).ToArray();
            
            for (int i = 0, size = length.Item1 * length.Item2; i < us.Length; i++) {
                vs.BlockCopyTo(us[i], i * size);
            }

            return us;
        }

        /// <summary>4次元配列化</summary>
        public static T[,,,] To4DArray<T>(this T[] vs, (int, int, int, int) length) where T : struct, IComparable {
            if (vs.Length != length.Item1 * length.Item2 * length.Item3 * length.Item4) {
                throw new ArgumentException(nameof(length));
            }

            T[,,,] us = new T[length.Item4, length.Item3, length.Item2, length.Item1];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>3次元配列配列化</summary>
        public static T[][,,] To3DArrays<T>(this T[] vs, (int, int, int) length, int batch) where T : struct, IComparable {
            if (vs.Length != length.Item1 * length.Item2 * length.Item3 * batch) {
                throw new ArgumentException(nameof(length));
            }

            T[][,,] us = (new T[batch][,,]).Select((_) => new T[length.Item3, length.Item2, length.Item1]).ToArray();
            
            for (int i = 0, size = length.Item1 * length.Item2 * length.Item3; i < us.Length; i++) {
                vs.BlockCopyTo(us[i], i * size);
            }

            return us;
        }

        /// <summary>5次元配列化</summary>
        public static T[,,,,] To5DArray<T>(this T[] vs, (int, int, int, int, int) length) where T : struct, IComparable {
            if (vs.Length != length.Item1 * length.Item2 * length.Item3 * length.Item4 * length.Item5) {
                throw new ArgumentException(nameof(length));
            }

            T[,,,,] us = new T[length.Item5, length.Item4, length.Item3, length.Item2, length.Item1];
            vs.BlockCopyTo(us, 0);

            return us;
        }

        /// <summary>4次元配列配列化</summary>
        public static T[][,,,] To4DArrays<T>(this T[] vs, (int, int, int, int) length, int batch) where T : struct, IComparable {
            if (vs.Length != length.Item1 * length.Item2 * length.Item3 * length.Item4 * batch) {
                throw new ArgumentException(nameof(length));
            }

            T[][,,,] us = (new T[batch][,,,]).Select((_) => new T[length.Item4, length.Item3, length.Item2, length.Item1]).ToArray();
            
            for (int i = 0, size = length.Item1 * length.Item2 * length.Item3 * length.Item4; i < us.Length; i++) {
                vs.BlockCopyTo(us[i], i * size);
            }

            return us;
        }
    }
}
