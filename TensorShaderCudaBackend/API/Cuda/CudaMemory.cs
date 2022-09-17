using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    using size_t = Int64;

    public static partial class Cuda {

        /// <summary>メモリ領域操作</summary>
        public static class Memory {

            /// <summary>デバイスメモリ領域確保</summary>
            internal static IntPtr Allocate<T>(ulong count) {
                IntPtr ptr = IntPtr.Zero;

                long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                ErrorCode result = NativeMethods.CudaMalloc.AsDelegate().Invoke(ref ptr, bytesize);
                if (result == ErrorCode.ErrorMemoryAllocation) {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();

#if DEBUG
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] Called finalizers");
#endif

                    result = NativeMethods.CudaMalloc.AsDelegate().Invoke(ref ptr, bytesize);
                }
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                return ptr;
            }

            /// <summary>デバイスメモリ領域解放</summary>
            internal static void Free(ref IntPtr ptr) {
                if (ptr == IntPtr.Zero) {
                    return;
                }

                ErrorCode result = NativeMethods.CudaFree.AsDelegate().Invoke(ptr);
                ptr = IntPtr.Zero;

                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }
            }

            /// <summary>ホストデバイス間メモリ領域コピー</summary>
            private static void Copy(IntPtr src_ptr, IntPtr dst_ptr, size_t count, cudaMemcpyKind kind) {
                try {
                    if (src_ptr == IntPtr.Zero) {
                        throw new ArgumentException(null, nameof(src_ptr));
                    }
                    if (dst_ptr == IntPtr.Zero) {
                        throw new ArgumentException(null, nameof(dst_ptr));
                    }

                    ErrorCode result = NativeMethods.CudaMemcpy.AsDelegate().Invoke(dst_ptr, src_ptr, count, kind);
                    if (result != ErrorCode.Success) {
                        throw new CudaException(result);
                    }
                }
                catch (AccessViolationException) {
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>ホストデバイス間メモリ領域コピー</summary>
            internal static void CopyDeviceToHost<T>(IntPtr src_ptr, T[] dst, ulong count, ulong index = 0) where T : struct, IComparable {
                if (src_ptr == IntPtr.Zero) {
                    throw new ArgumentException(null, nameof(src_ptr));
                }
                if (dst is null || count + index > (ulong)dst.LongLength) {
                    throw new ArgumentException(null, nameof(dst));
                }

                long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);
                GCHandle pinned_handle = GCHandle.Alloc(dst, GCHandleType.Pinned);

                try {
                    IntPtr dst_ptr = Marshal.UnsafeAddrOfPinnedArrayElement(dst, checked((int)index));
                    Copy(src_ptr, dst_ptr, bytesize, cudaMemcpyKind.DeviceToHost);
                }
                finally {
                    pinned_handle.Free();
                }
            }

            /// <summary>ホストデバイス間メモリ領域コピー</summary>
            internal static void CopyHostToDevice<T>(T[] src, IntPtr dst_ptr, ulong count, ulong index = 0) where T : struct, IComparable {
                if (src is null || count + index > (ulong)src.LongLength) {
                    throw new ArgumentException(null, nameof(src));
                }
                if (dst_ptr == IntPtr.Zero) {
                    throw new ArgumentException(null, nameof(dst_ptr));
                }

                long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);
                GCHandle pinned_handle = GCHandle.Alloc(src, GCHandleType.Pinned);

                try {
                    IntPtr src_ptr = Marshal.UnsafeAddrOfPinnedArrayElement(src, checked((int)index));
                    Copy(src_ptr, dst_ptr, bytesize, cudaMemcpyKind.HostToDevice);
                }
                finally {
                    pinned_handle.Free();
                }
            }

            /// <summary>デバイス間メモリ領域コピー</summary>
            internal static void CopyDeviceToDevice<T>(IntPtr src_ptr, IntPtr dst_ptr, ulong count) where T : struct, IComparable {
                if (src_ptr == IntPtr.Zero) {
                    throw new ArgumentException(null, nameof(src_ptr));
                }
                if (dst_ptr == IntPtr.Zero) {
                    throw new ArgumentException(null, nameof(dst_ptr));
                }

                long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                Copy(src_ptr, dst_ptr, bytesize, cudaMemcpyKind.DeviceToDevice);
            }

            /// <summary>非同期ホストデバイス間メモリ領域コピー</summary>
            private static void CopyAsync(IntPtr src_ptr, IntPtr dst_ptr, size_t count, cudaMemcpyKind kind, IntPtr stream) {
                try {
                    if (src_ptr == IntPtr.Zero) {
                        throw new ArgumentException(null, nameof(src_ptr));
                    }
                    if (dst_ptr == IntPtr.Zero) {
                        throw new ArgumentException(null, nameof(dst_ptr));
                    }
                    if (stream == IntPtr.Zero) {
                        throw new ArgumentException(null, nameof(stream));
                    }

                    ErrorCode result = NativeMethods.CudaMemcpyAsync.AsDelegate().Invoke(dst_ptr, src_ptr, count, kind, stream);
                    if (result != ErrorCode.Success) {
                        throw new CudaException(result);
                    }
                }
                catch (AccessViolationException) {
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>非同期デバイス間メモリ領域コピー</summary>
            internal static void CopyDeviceToDeviceAsync<T>(IntPtr src_ptr, IntPtr dst_ptr, ulong count, IntPtr stream) where T : struct, IComparable {
                if (src_ptr == IntPtr.Zero) {
                    throw new ArgumentException(null, nameof(src_ptr));
                }
                if (dst_ptr == IntPtr.Zero) {
                    throw new ArgumentException(null, nameof(dst_ptr));
                }

                long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                CopyAsync(src_ptr, dst_ptr, bytesize, cudaMemcpyKind.DeviceToDevice, stream);
            }

            /// <summary>デバイスメモリゼロクリア</summary>
            internal static void Zeroset<T>(IntPtr ptr, ulong count) where T : struct, IComparable {
                if (ptr == IntPtr.Zero) {
                    throw new ArgumentException(null, nameof(ptr));
                }

                long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                ErrorCode result = NativeMethods.CudaMemset.AsDelegate().Invoke(ptr, 0, bytesize);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }
            }

            /// <summary>非同期デバイスメモリゼロクリア</summary>
            internal static void ZerosetAsync<T>(IntPtr ptr, ulong count, IntPtr stream) where T : struct, IComparable {
                if (ptr == IntPtr.Zero) {
                    throw new ArgumentException(null, nameof(ptr));
                }
                if (stream == IntPtr.Zero) {
                    throw new ArgumentException(null, nameof(stream));
                }

                long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                ErrorCode result = NativeMethods.CudaMemsetAsync.AsDelegate().Invoke(ptr, 0, bytesize, stream);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }
            }
        }
    }
}
