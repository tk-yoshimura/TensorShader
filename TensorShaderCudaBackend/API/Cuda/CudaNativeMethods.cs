﻿using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    using size_t = Int64;
    using cudaError_t = Cuda.ErrorCode;
    using cudaDeviceProp = Cuda.DeviceProp;

    public static partial class Cuda {

        #pragma warning disable IDE1006 // 命名スタイル
        private static class NativeMethods {
            const string DllName = "cudart64_101.dll";

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaDeviceReset();

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaDeviceSynchronize();

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaSetDevice(int device);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaGetDevice(ref int device);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaGetDeviceCount(ref int count);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaSetDeviceFlags(uint flags);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaGetDeviceFlags(ref uint flags);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaGetDeviceProperties(ref cudaDeviceProp prop, int device);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaFree(IntPtr devPtr);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaMalloc(ref IntPtr devPtr, size_t size);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaMemcpy(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaMemcpyAsync(IntPtr dst, IntPtr src, size_t count, cudaMemcpyKind kind, IntPtr stream);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaMemset(IntPtr devPtr, int value, size_t count);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaMemsetAsync(IntPtr devPtr, int value, size_t count, IntPtr stream);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaDriverGetVersion(ref int driverVersion);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaRuntimeGetVersion(ref int runtimeVersion);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaStreamCreate(ref IntPtr pStream);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaStreamDestroy(IntPtr stream);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaStreamSynchronize(IntPtr stream);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaMemGetInfo(ref size_t free, ref size_t total);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
            public static extern cudaError_t cudaProfilerInitialize(string configFile, string outputFile, cudaOutputMode outputMode);

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaProfilerStart();

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern cudaError_t cudaProfilerStop();

            [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
            public static extern IntPtr cudaGetErrorString(cudaError_t error);
        }
        #pragma warning restore IDE1006 // 命名スタイル
    }
}
