using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {

    /// <summary>NvCuda API</summary>
    public static partial class Nvcuda {

        static volatile bool initialized = false;

        /// <summary>ドライバの初期化</summary>
        public static void InitDriver() {
            ResultCode result = NativeMethods.cuInit(0);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            /* デバイスメモリ確保チェック */
            IntPtr ptr = Cuda.Memory.Allocate<int>(4);
            Cuda.Memory.Free(ref ptr);

            initialized = true;
        }

        /// <summary>PTXからシェーダモジュールをロード</summary>
        internal static IntPtr LoadModule(IntPtr ptx) {
            if (!initialized) InitDriver();

            IntPtr module = IntPtr.Zero;

            ResultCode result = NativeMethods.cuModuleLoadData(ref module, ptx);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            return module;
        }

        /// <summary>シェーダモジュールをアンロード</summary>
        internal static void UnloadModule(ref IntPtr module) {
            if (module == IntPtr.Zero) {
                return;
            }

            ResultCode result = NativeMethods.cuModuleUnload(module);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            module = IntPtr.Zero;
        }

        /// <summary>モジュールからカーネル取得</summary>
        internal static IntPtr GetKernel(IntPtr module, string entrypoint) {
            IntPtr kernel = IntPtr.Zero;

            ResultCode result = NativeMethods.cuModuleGetFunction(ref kernel, module, entrypoint);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            return kernel;
        }

        /// <summary>モジュールからグローバルポインタ取得</summary>
        internal static (IntPtr ptr, Int64 size) GetGlobalPointer(IntPtr module, string name) {
            IntPtr ptr = IntPtr.Zero;
            Int64 size = 0;

            ResultCode result = NativeMethods.cuModuleGetGlobal_v2(ref ptr, ref size, module, name);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            return (ptr, size);
        }

        /// <summary>カーネルを実行</summary>
        internal static void LaunchKernel(IntPtr kernel, uint grid_dimx, uint grid_dimy, uint grid_dimz, uint block_dimx, uint block_dimy, uint block_dimz, uint dynamic_shared_memory_bytes, IntPtr stream, IntPtr kernel_params, IntPtr extra) {
            ResultCode result = NativeMethods.cuLaunchKernel(
                kernel,
                grid_dimx, grid_dimy, grid_dimz, block_dimx, block_dimy, block_dimz,
                dynamic_shared_memory_bytes, stream, kernel_params, extra
            );
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }
        }

        /// <summary>エラーコードメッセージ</summary>
        internal static string GetErrorString(ResultCode error) {
            IntPtr ptr = IntPtr.Zero;
            NativeMethods.cuGetErrorString(error, ref ptr);

            return Marshal.PtrToStringAnsi(ptr);
        }
    }
}
