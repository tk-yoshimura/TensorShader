using System;
using System.Runtime.InteropServices;

using TensorShaderCudaBackend.Dll;

namespace TensorShaderCudaBackend.API {
    using cudaDeviceProp = Cuda.DeviceProp;
    using cudaError_t = Cuda.ErrorCode;
    using size_t = Int64;

    public static partial class Cuda {

#pragma warning disable IDE1006
        private static class NativeMethods {

            static readonly NativeDll dll = CudaDll.Cuda;

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaDeviceReset();
            public static NativeMethod<cudaDeviceReset> CudaDeviceReset { get; }
                = new NativeMethod<cudaDeviceReset>(dll, nameof(cudaDeviceReset));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaDeviceSynchronize();
            public static NativeMethod<cudaDeviceSynchronize> CudaDeviceSynchronize { get; }
                = new NativeMethod<cudaDeviceSynchronize>(dll, nameof(cudaDeviceSynchronize));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaSetDevice(int device);
            public static NativeMethod<cudaSetDevice> CudaSetDevice { get; }
                = new NativeMethod<cudaSetDevice>(dll, nameof(cudaSetDevice));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaGetDevice(ref int device);
            public static NativeMethod<cudaGetDevice> CudaGetDevice { get; }
                = new NativeMethod<cudaGetDevice>(dll, nameof(cudaGetDevice));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaGetDeviceCount(ref int count);
            public static NativeMethod<cudaGetDeviceCount> CudaGetDeviceCount { get; }
                = new NativeMethod<cudaGetDeviceCount>(dll, nameof(cudaGetDeviceCount));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaSetDeviceFlags(uint flags);
            public static NativeMethod<cudaSetDeviceFlags> CudaSetDeviceFlags { get; }
                = new NativeMethod<cudaSetDeviceFlags>(dll, nameof(cudaSetDeviceFlags));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaGetDeviceFlags(ref uint flags);
            public static NativeMethod<cudaGetDeviceFlags> CudaGetDeviceFlags { get; }
                = new NativeMethod<cudaGetDeviceFlags>(dll, nameof(cudaGetDeviceFlags));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaGetDeviceProperties(ref cudaDeviceProp prop, int device);
            public static NativeMethod<cudaGetDeviceProperties> CudaGetDeviceProperties { get; }
                = new NativeMethod<cudaGetDeviceProperties>(dll, nameof(cudaGetDeviceProperties));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaFree(IntPtr devPtr);
            public static NativeMethod<cudaFree> CudaFree { get; }
                = new NativeMethod<cudaFree>(dll, nameof(cudaFree));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaMalloc(ref IntPtr devPtr, size_t size);
            public static NativeMethod<cudaMalloc> CudaMalloc { get; }
                = new NativeMethod<cudaMalloc>(dll, nameof(cudaMalloc));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaMemcpy(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind);
            public static NativeMethod<cudaMemcpy> CudaMemcpy { get; }
                = new NativeMethod<cudaMemcpy>(dll, nameof(cudaMemcpy));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaMemcpyAsync(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind, IntPtr stream);
            public static NativeMethod<cudaMemcpyAsync> CudaMemcpyAsync { get; }
                = new NativeMethod<cudaMemcpyAsync>(dll, nameof(cudaMemcpyAsync));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaMemset(IntPtr devPtr, int value, size_t count);
            public static NativeMethod<cudaMemset> CudaMemset { get; }
                = new NativeMethod<cudaMemset>(dll, nameof(cudaMemset));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaMemsetAsync(IntPtr devPtr, int value, size_t count, IntPtr stream);
            public static NativeMethod<cudaMemsetAsync> CudaMemsetAsync { get; }
                = new NativeMethod<cudaMemsetAsync>(dll, nameof(cudaMemsetAsync));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaDriverGetVersion(ref int driverVersion);
            public static NativeMethod<cudaDriverGetVersion> CudaDriverGetVersion { get; }
                = new NativeMethod<cudaDriverGetVersion>(dll, nameof(cudaDriverGetVersion));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaRuntimeGetVersion(ref int runtimeVersion);
            public static NativeMethod<cudaRuntimeGetVersion> CudaRuntimeGetVersion { get; }
                = new NativeMethod<cudaRuntimeGetVersion>(dll, nameof(cudaRuntimeGetVersion));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaStreamCreate(ref IntPtr pStream);
            public static NativeMethod<cudaStreamCreate> CudaStreamCreate { get; }
                = new NativeMethod<cudaStreamCreate>(dll, nameof(cudaStreamCreate));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaStreamDestroy(IntPtr stream);
            public static NativeMethod<cudaStreamDestroy> CudaStreamDestroy { get; }
                = new NativeMethod<cudaStreamDestroy>(dll, nameof(cudaStreamDestroy));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaStreamSynchronize(IntPtr stream);
            public static NativeMethod<cudaStreamSynchronize> CudaStreamSynchronize { get; }
                = new NativeMethod<cudaStreamSynchronize>(dll, nameof(cudaStreamSynchronize));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaMemGetInfo(ref size_t free, ref size_t total);
            public static NativeMethod<cudaMemGetInfo> CudaMemGetInfo { get; }
                = new NativeMethod<cudaMemGetInfo>(dll, nameof(cudaMemGetInfo));

            [UnmanagedFunctionPointer(CallingConvention.StdCall, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public delegate cudaError_t cudaProfilerInitialize(string configFile, string outputFile, cudaOutputMode outputMode);
            public static NativeMethod<cudaProfilerInitialize> CudaProfilerInitialize { get; }
                = new NativeMethod<cudaProfilerInitialize>(dll, nameof(cudaProfilerInitialize));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaProfilerStart();
            public static NativeMethod<cudaProfilerStart> CudaProfilerStart { get; }
                = new NativeMethod<cudaProfilerStart>(dll, nameof(cudaProfilerStart));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate cudaError_t cudaProfilerStop();
            public static NativeMethod<cudaProfilerStop> CudaProfilerStop { get; }
                = new NativeMethod<cudaProfilerStop>(dll, nameof(cudaProfilerStop));

            [UnmanagedFunctionPointer(CallingConvention.StdCall)]
            public delegate IntPtr cudaGetErrorString(cudaError_t error);
            public static NativeMethod<cudaGetErrorString> CudaGetErrorString { get; }
                = new NativeMethod<cudaGetErrorString>(dll, nameof(cudaGetErrorString));
        }
#pragma warning restore IDE1006
    }
}
