using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.API {
    using cudaDeviceProp = Cuda.DeviceProp;

    /// <summary>CUDA API</summary>
    public static partial class Cuda {

        /// <summary>カレントデバイスID</summary>
        private static int device_id = -1;

        /// <summary>カレントデバイスプロパティ</summary>
        public static DeviceProp CurrectDeviceProperty { private set; get; } = DeviceProperty(CurrectDeviceID);

        /// <summary>デバイスID</summary>
        public static int CurrectDeviceID {
            get {
                if (Cuda.device_id >= 0) {
                    return Cuda.device_id;
                }

                int device_id = 0;

                ErrorCode result = NativeMethods.CudaGetDevice.AsDelegate().Invoke(ref device_id);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                return Cuda.device_id = device_id;
            }

            set {
                if (value == CurrectDeviceID) {
                    return;
                }

                ErrorCode result = NativeMethods.CudaSetDevice.AsDelegate().Invoke(value);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                Cuda.device_id = value;
                Cuda.CurrectDeviceProperty = DeviceProperty(CurrectDeviceID);
            }
        }

        /// <summary>CUDA対応デバイス数</summary>
        public static int DeviceCounts {
            get {
                int count = 0;

                ErrorCode result = NativeMethods.CudaGetDeviceCount.AsDelegate().Invoke(ref count);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                return count;
            }
        }

        /// <summary>デバイスフラグ</summary>
        public static uint DeviceFlags {
            get {
                uint flags = 0;

                ErrorCode result = NativeMethods.CudaGetDeviceFlags.AsDelegate().Invoke(ref flags);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                return flags;
            }

            set {
                ErrorCode result = NativeMethods.CudaSetDeviceFlags.AsDelegate().Invoke(value);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }
            }
        }

        /// <summary>デバイスプロパティ</summary>
        public static DeviceProp DeviceProperty(int device_id) {
            cudaDeviceProp prop = new();

            ErrorCode result = NativeMethods.CudaGetDeviceProperties.AsDelegate().Invoke(ref prop, device_id);
            if (result != ErrorCode.Success) {
                throw new CudaException(result);
            }

            return prop;
        }

        /// <summary>ドライババージョン</summary>
        public static int DriverVersion {
            get {
                int version = 0;

                ErrorCode result = NativeMethods.CudaDriverGetVersion.AsDelegate().Invoke(ref version);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                return version;
            }
        }

        /// <summary>ランタイムバージョン</summary>
        public static int RuntimeVersion {
            get {
                int version = 0;

                ErrorCode result = NativeMethods.CudaRuntimeGetVersion.AsDelegate().Invoke(ref version);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                return version;
            }
        }

        /// <summary>デバイスメモリ容量</summary>
        public static (long total, long free) MemoryInfo {
            get {
                long total = 0, free = 0;

                ErrorCode result = NativeMethods.CudaMemGetInfo.AsDelegate().Invoke(ref free, ref total);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                return (total, free);
            }
        }

        /// <summary>デバイスメモリ使用量</summary>
        public static double MemoryUsage {
            get {
                long total = 0, free = 0;

                ErrorCode result = NativeMethods.CudaMemGetInfo.AsDelegate().Invoke(ref free, ref total);
                if (result != ErrorCode.Success) {
                    throw new CudaException(result);
                }

                return (double)(total - free) / total;
            }
        }

        /// <summary>実行中のカーネル終了まで待機</summary>
        public static void Synchronize() {
            ErrorCode result = NativeMethods.CudaDeviceSynchronize.AsDelegate().Invoke();
            if (result != ErrorCode.Success) {
                throw new CudaException(result);
            }
        }

        /// <summary>デバイスをリセット</summary>
        public static void Reset() {
            ErrorCode result = NativeMethods.CudaDeviceReset.AsDelegate().Invoke();
            if (result != ErrorCode.Success) {
                throw new CudaException(result);
            }
        }

        /// <summary>エラーコードメッセージ</summary>
        internal static string GetErrorString(ErrorCode error) {
            IntPtr ptr = NativeMethods.CudaGetErrorString.AsDelegate().Invoke(error);

            string str = string.Empty;
            if (ptr != IntPtr.Zero) {
                str = Marshal.PtrToStringAnsi(ptr);
            }

            return str;
        }
    }
}
