using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {

    /// <summary>NvCuda API</summary>
    public static partial class Nvcuda {

        static volatile bool initialized = false;

        /// <summary>ドライバの初期化</summary>
        public static void InitDriver() {
            ResultCode result = NativeMethods.CuInit.AsDelegate().Invoke(0);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            /* デバイスメモリ確保チェック */
            IntPtr ptr = Cuda.Memory.Allocate<int>(4);
            Cuda.Memory.Free(ref ptr);

            /* キャッシュ配分比変更 */
            SetCacheConfig(FuncCache.PreferL1);

            initialized = true;
        }

        /// <summary>PTXからシェーダモジュールをロード</summary>
        internal static IntPtr LoadModule(IntPtr ptx) {
            if (!initialized) InitDriver();

            IntPtr module = IntPtr.Zero;

            ResultCode result = NativeMethods.CuModuleLoadData.AsDelegate().Invoke(ref module, ptx);
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

            ResultCode result = NativeMethods.CuModuleUnload.AsDelegate().Invoke(module);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            module = IntPtr.Zero;
        }

        /// <summary>モジュールからカーネル取得</summary>
        internal static IntPtr GetKernel(IntPtr module, string entrypoint) {
            IntPtr kernel = IntPtr.Zero;

            ResultCode result = NativeMethods.CuModuleGetFunction.AsDelegate().Invoke(ref kernel, module, entrypoint);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            return kernel;
        }

        /// <summary>モジュールからグローバルポインタ取得</summary>
        internal static (IntPtr ptr, Int64 size) GetGlobalPointer(IntPtr module, string name) {
            IntPtr ptr = IntPtr.Zero;
            Int64 size = 0;

            ResultCode result = NativeMethods.CuModuleGetGlobal_v2.AsDelegate().Invoke(ref ptr, ref size, module, name);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }

            return (ptr, size);
        }

        /// <summary>共有メモリ/L1キャッシュ配分比を変更</summary>
        internal static void SetCacheConfig(FuncCache func_cache) {
            ResultCode result = NativeMethods.CuCtxSetCacheConfig.AsDelegate().Invoke(func_cache);
            if (result != ResultCode.Success) {
                throw new CudaException(result);
            }
        }

        /// <summary>カーネルを実行</summary>
        internal static void LaunchKernel(IntPtr kernel, uint grid_dimx, uint grid_dimy, uint grid_dimz, uint block_dimx, uint block_dimy, uint block_dimz, uint dynamic_shared_memory_bytes, IntPtr stream, IntPtr kernel_params, IntPtr extra) {
            ResultCode result = NativeMethods.CuLaunchKernel.AsDelegate().Invoke(
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
            NativeMethods.CuGetErrorString.AsDelegate().Invoke(error, ref ptr);

            string str = string.Empty;
            if (ptr != IntPtr.Zero) {
                str = Marshal.PtrToStringAnsi(ptr);
                Marshal.FreeHGlobal(ptr);
            }

            return str;
        }
    }
}
