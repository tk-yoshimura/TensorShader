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
            [HandleProcessCorruptedStateExceptions]
            internal static IntPtr Allocate<T>(ulong count) {
                try{
                    IntPtr ptr = IntPtr.Zero;

                    long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                    ErrorCode result = NativeMethods.cudaMalloc(ref ptr, bytesize);
                    if (result == ErrorCode.ErrorMemoryAllocation) {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();

#if DEBUG
                        Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] Called finalizers");
#endif

                        result = NativeMethods.cudaMalloc(ref ptr, bytesize);
                    }
                    if (result != ErrorCode.Success) {
                        throw new CudaException(result);
                    }

                    return ptr;
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>デバイスメモリ領域解放</summary>
            [HandleProcessCorruptedStateExceptions]
            internal static void Free(ref IntPtr ptr) {
                try{
                    if (ptr == IntPtr.Zero) {
                        return;
                    }

                    ErrorCode result = NativeMethods.cudaFree(ptr);
                    ptr = IntPtr.Zero;

                    if (result != ErrorCode.Success) {
                        throw new CudaException(result);
                    }
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>ホストデバイス間メモリ領域コピー</summary>
            [HandleProcessCorruptedStateExceptions]
            private static void Copy(IntPtr src_ptr, IntPtr dst_ptr, size_t count, cudaMemcpyKind kind) {
                try { 
                    if (src_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(src_ptr));
                    }
                    if (dst_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(dst_ptr));
                    }

                    ErrorCode result = NativeMethods.cudaMemcpy(dst_ptr, src_ptr, count, kind);
                    if (result != ErrorCode.Success) {
                        throw new CudaException(result);
                    }
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>ホストデバイス間メモリ領域コピー</summary>
            [HandleProcessCorruptedStateExceptions]
            internal static void CopyDeviceToHost<T>(IntPtr src_ptr, T[] dst, ulong count) where T : struct, IComparable {
                try { 
                    if (src_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(src_ptr));
                    }
                    if (dst == null || count > (ulong)dst.LongLength) {
                        throw new ArgumentException(nameof(dst));
                    }

                    long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);
                    GCHandle pinned_handle = GCHandle.Alloc(dst, GCHandleType.Pinned);

                    try { 
                        IntPtr dst_ptr = Marshal.UnsafeAddrOfPinnedArrayElement(dst, 0);
                        Copy(src_ptr, dst_ptr, bytesize, cudaMemcpyKind.DeviceToHost);
                    }
                    finally { 
                        pinned_handle.Free();
                    }
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>ホストデバイス間メモリ領域コピー</summary>
            [HandleProcessCorruptedStateExceptions]
            internal static void CopyHostToDevice<T>(T[] src, IntPtr dst_ptr, ulong count) where T : struct, IComparable {
                try { 
                    if (src == null || count > (ulong)src.LongLength) {
                        throw new ArgumentException(nameof(src));
                    }
                    if (dst_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(dst_ptr));
                    }

                    long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);
                    GCHandle pinned_handle = GCHandle.Alloc(src, GCHandleType.Pinned);

                    try { 
                        IntPtr src_ptr = Marshal.UnsafeAddrOfPinnedArrayElement(src, 0);
                        Copy(src_ptr, dst_ptr, bytesize, cudaMemcpyKind.HostToDevice);
                    }
                    finally { 
                        pinned_handle.Free();
                    }
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>デバイス間メモリ領域コピー</summary>
            [HandleProcessCorruptedStateExceptions]
            internal static void CopyDeviceToDevice<T>(IntPtr src_ptr, IntPtr dst_ptr, ulong count) where T : struct, IComparable {
                try { 
                    if (src_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(src_ptr));
                    }
                    if (dst_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(dst_ptr));
                    }

                    long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                    Copy(src_ptr, dst_ptr, bytesize, cudaMemcpyKind.DeviceToDevice);
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>非同期ホストデバイス間メモリ領域コピー</summary>
            [HandleProcessCorruptedStateExceptions]
            private static void CopyAsync(IntPtr src_ptr, IntPtr dst_ptr, size_t count, cudaMemcpyKind kind, IntPtr stream) {
                try { 
                    if (src_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(src_ptr));
                    }
                    if (dst_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(dst_ptr));
                    }
                    if (stream == IntPtr.Zero) {
                        throw new ArgumentException(nameof(stream));
                    }

                    ErrorCode result = NativeMethods.cudaMemcpyAsync(dst_ptr, src_ptr, count, kind, stream);
                    if (result != ErrorCode.Success) {
                        throw new CudaException(result);
                    }
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>非同期デバイス間メモリ領域コピー</summary>
            [HandleProcessCorruptedStateExceptions]
            internal static void CopyDeviceToDeviceAsync<T>(IntPtr src_ptr, IntPtr dst_ptr, ulong count, IntPtr stream) where T : struct, IComparable {
                try { 
                    if (src_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(src_ptr));
                    }
                    if (dst_ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(dst_ptr));
                    }

                    long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                    CopyAsync(src_ptr, dst_ptr, bytesize, cudaMemcpyKind.DeviceToDevice, stream);
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>デバイスメモリゼロクリア</summary>
            [HandleProcessCorruptedStateExceptions]
            internal static void Zeroset<T>(IntPtr ptr, ulong count) where T : struct, IComparable {
                try { 
                    if (ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(ptr));
                    }

                    long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                    ErrorCode result = NativeMethods.cudaMemset(ptr, 0, bytesize);
                    if (result != ErrorCode.Success) {
                        throw new CudaException(result);
                    }
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }

            /// <summary>非同期デバイスメモリゼロクリア</summary>
            [HandleProcessCorruptedStateExceptions]
            internal static void ZerosetAsync<T>(IntPtr ptr, ulong count, IntPtr stream) where T : struct, IComparable {
                try { 
                    if (ptr == IntPtr.Zero) {
                        throw new ArgumentException(nameof(ptr));
                    }
                    if (stream == IntPtr.Zero) {
                        throw new ArgumentException(nameof(stream));
                    }

                    long bytesize = (long)((ulong)Marshal.SizeOf(typeof(T)) * count);

                    ErrorCode result = NativeMethods.cudaMemsetAsync(ptr, 0, bytesize, stream);
                    if (result != ErrorCode.Success) {
                        throw new CudaException(result);
                    }
                }
                catch (AccessViolationException){
                    Trace.WriteLine($"[{typeof(Memory).Name}.{MethodBase.GetCurrentMethod().Name}] AccessViolationException");
                    throw;
                }
            }
        }
    }
}
