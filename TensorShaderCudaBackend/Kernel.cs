using System;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;

using TensorShaderCudaBackend.API;

namespace TensorShaderCudaBackend {

    /// <summary>カーネル</summary>
    public sealed partial class Kernel : IDisposable {

        private IntPtr module = IntPtr.Zero, kernel = IntPtr.Zero;

        /// <summary>有効か</summary>
        public bool IsValid => module != IntPtr.Zero && kernel != IntPtr.Zero;

        /// <summary>概要</summary>
        public string Overview { private set; get; }

        /// <summary>最大ブロックサイズ</summary>
        /// <remarks>
        /// = 512 (constant)
        /// Fermi アーキテクチャ(2009)の最大ブロックサイズ および cc7.5のcuda memsetのブロックサイズに準拠
        /// </remarks>
        public static uint MaxBlockSize => Math.Min(512, (uint)Cuda.CurrectDeviceProperty.MaxThreadsPerBlock);

        /// <summary>命令並列数</summary>
        /// <remarks>
        /// = 16 (constant)
        /// </remarks>
        public static uint HalfWarp = 16;

        /// <summary>コンストラクタ</summary>
        /// <param name="code">コード</param>
        /// <param name="entrypoint">関数名</param>
        /// <param name="debug">デバッグ情報を生成するか</param>
        public Kernel(string code, string entrypoint, bool debug = false) {

            string[] options = debug
                               ? new string[] { Nvrtc.CompileOptions.ArchitectureTarget, Nvrtc.CompileOptions.Debug }
                               : new string[] { Nvrtc.CompileOptions.ArchitectureTarget };

            string ptx = Nvrtc.CompileProgram(code, entrypoint + ".cu", options);
            IntPtr ptx_ansi = Marshal.StringToHGlobalAnsi(ptx);

            try {
                this.module = Nvcuda.LoadModule(ptx_ansi);
                this.kernel = Nvcuda.GetKernel(module, entrypoint);
            }
            finally {
                Marshal.FreeHGlobal(ptx_ansi);
            }

            this.Overview = entrypoint;
        }

        /// <summary>定数メモリへストア</summary>
        public void StoreConstMemory<T>(string name, T[] array) where T : struct, IComparable {
            StoreConstMemory(name, array, (uint)array.Length);
        }

        /// <summary>定数メモリへストア</summary>
        public void StoreConstMemory<T>(string name, T[] array, uint count) where T : struct, IComparable {
            if (count > array.Length) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            if (!IsValid) {
                throw new ObjectDisposedException(GetType().FullName);
            }

            (IntPtr ptr, Int64 size) = Nvcuda.GetGlobalPointer(module, name);

            if (size < count * Marshal.SizeOf(typeof(T))) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            Cuda.Memory.CopyHostToDevice(array, ptr, (ulong)array.Length);
        }

        /// <summary>定数メモリへストア</summary>
        public void StoreConstMemory<T>(string name, CudaArray<T> array) where T : struct, IComparable {
            StoreConstMemory(name, array, (uint)array.Length);
        }

        /// <summary>定数メモリへストア</summary>
        public void StoreConstMemory<T>(string name, CudaArray<T> array, uint count) where T : struct, IComparable {
            if (count > array.Length) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            if (!IsValid) {
                throw new ObjectDisposedException(GetType().FullName);
            }

            (IntPtr ptr, Int64 size) = Nvcuda.GetGlobalPointer(module, name);

            if (size < count * Marshal.SizeOf(typeof(T))) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            Cuda.Memory.CopyDeviceToDevice<T>(array.Ptr, ptr, count);
        }

        /// <summary>定数メモリへストア</summary>
        public void StoreConstMemoryAsync<T>(Stream stream, string name, CudaArray<T> array) where T : struct, IComparable {
            StoreConstMemoryAsync(stream, name, array, (uint)array.Length);
        }

        /// <summary>定数メモリへ非同期ストア</summary>
        public void StoreConstMemoryAsync<T>(Stream stream, string name, CudaArray<T> array, uint count) where T : struct, IComparable {
            if (count > array.Length) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            if (!IsValid) {
                throw new ObjectDisposedException(GetType().FullName);
            }

            (IntPtr ptr, Int64 size) = Nvcuda.GetGlobalPointer(module, name);

            if (size < count * Marshal.SizeOf(typeof(T))) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            Cuda.Memory.CopyDeviceToDeviceAsync<T>(array.Ptr, ptr, count, stream.Ptr);
        }

        /// <summary>実行</summary>
        /// <remarks>streamにnullを指定したときデフォルトストリームが使用される</remarks>
        public void Execute(uint indexes, uint dynamic_shared_memory_bytes, Stream stream, params object[] args) {
            Execute(indexes, DefaultBlockSize(indexes), dynamic_shared_memory_bytes, stream, args);
        }

        /// <summary>実行</summary>
        /// <remarks>streamにnullを指定したときデフォルトストリームが使用される</remarks>
        public void Execute((uint x, uint y) indexes, uint dynamic_shared_memory_bytes, Stream stream, params object[] args) {
            Execute(indexes, DefaultBlockSize(indexes), dynamic_shared_memory_bytes, stream, args);
        }

        /// <summary>実行</summary>
        /// <remarks>streamにnullを指定したときデフォルトストリームが使用される</remarks>
        public void Execute((uint x, uint y, uint z) indexes, uint dynamic_shared_memory_bytes, Stream stream, params object[] args) {
            Execute(indexes, DefaultBlockSize(indexes), dynamic_shared_memory_bytes, stream, args);
        }

        /// <summary>実行</summary>
        /// <remarks>streamにnullを指定したときデフォルトストリームが使用される</remarks>
        public void Execute(uint indexes, uint block, uint dynamic_shared_memory_bytes, Stream stream, params object[] args) {
            Execute((indexes, 1, 1), (block, 1, 1), dynamic_shared_memory_bytes, stream, args);
        }

        /// <summary>実行</summary>
        /// <remarks>streamにnullを指定したときデフォルトストリームが使用される</remarks>
        public void Execute((uint x, uint y) indexes, (uint x, uint y) block, uint dynamic_shared_memory_bytes, Stream stream, params object[] args) {
            Execute((indexes.x, indexes.y, 1), (block.x, block.y, 1), dynamic_shared_memory_bytes, stream, args);
        }

        /// <summary>実行</summary>
        /// <remarks>streamにnullを指定したときデフォルトストリームが使用される</remarks>
        public void Execute((uint x, uint y, uint z) indexes, (uint x, uint y, uint z) block, uint dynamic_shared_memory_bytes, Stream stream, params object[] args) {
            if (!IsValid) {
                throw new ObjectDisposedException(GetType().FullName);
            }

            (uint x, uint y, uint z) grid =
                ((indexes.x + block.x - 1) / block.x, (indexes.y + block.y - 1) / block.y, (indexes.z + block.z - 1) / block.z);

            using (Arguments arguments = new(args)) {
                Nvcuda.LaunchKernel(
                    kernel,
                    grid.x, grid.y, grid.z,
                    block.x, block.y, block.z,
                    dynamic_shared_memory_bytes,
                    stream is not null ? stream.Ptr : IntPtr.Zero,
                    arguments.Ptr, IntPtr.Zero
                );
            }
        }

        /// <summary>既定ブロック数</summary>
        /// <remarks>常に2の冪数</remarks>
        public static uint DefaultBlockSize(uint indexes) {
            if (indexes < 1) {
                throw new ArgumentException(nameof(indexes));
            }

            const double occupancy = 0.8; /*スレッドの稼働率下限*/

            for (uint block = MaxBlockSize; block > HalfWarp; block /= 2) {
                uint threads = (indexes + block - 1) / block * block;

                if (indexes > threads * occupancy) {
                    return block;
                }
            }

            for (uint block = 1; block <= HalfWarp; block *= 2) {
                if (indexes <= block) {
                    return block;
                }
            }

            return HalfWarp;
        }

        /// <summary>既定ブロック数</summary>
        /// <remarks>常に2の冪数</remarks>
        public static (uint x, uint y) DefaultBlockSize((uint x, uint y) indexes) {
            uint block_x = DefaultBlockSize(indexes.x);
            uint block_y = Math.Min(Math.Min(MaxBlockSize / block_x, DefaultBlockSize(indexes.y)),
                                    (uint)Cuda.CurrectDeviceProperty.MaxThreadsDim[1]);

            return (block_x, block_y);
        }

        /// <summary>既定ブロック数</summary>
        /// <remarks>常に2の冪数</remarks>
        public static (uint x, uint y, uint z) DefaultBlockSize((uint x, uint y, uint z) indexes) {
            uint block_x = DefaultBlockSize(indexes.x);
            uint block_y = Math.Min(Math.Min(MaxBlockSize / block_x, DefaultBlockSize(indexes.y)),
                                    (uint)Cuda.CurrectDeviceProperty.MaxThreadsDim[1]);
            uint block_z = Math.Min(Math.Min(MaxBlockSize / (block_x * block_y), DefaultBlockSize(indexes.z)),
                                    (uint)Cuda.CurrectDeviceProperty.MaxThreadsDim[2]);

            return (block_x, block_y, block_z);
        }

        /// <summary>グリッド数が最小化となるブロック数</summary>
        /// <remarks>常に2の冪数</remarks>
        public static (uint x, uint y) MinimizeGridsBlockSize((uint x, uint y) indexes) {
            if (indexes.x < 1 || indexes.y < 1) {
                throw new ArgumentException(nameof(indexes));
            }

            uint block_x = 1, block_y = 1;

            while (block_x < indexes.x && block_x < MaxBlockSize) {
                block_x *= 2;
            }

            while (block_y < indexes.y && block_y < MaxBlockSize) {
                block_y *= 2;
            }

            while (block_x * block_y > MaxBlockSize) {
                if (block_x * indexes.y > block_y * indexes.x) {
                    block_x /= 2;
                }
                else {
                    block_y /= 2;
                }
            }

            return (block_x, block_y);
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return $"{nameof(Kernel)} {Overview}";
        }

        /// <summary>破棄</summary>
        public void Dispose() {
            Nvcuda.UnloadModule(ref module);
            kernel = IntPtr.Zero;
            Overview = null;

            GC.SuppressFinalize(this);

#if DEBUG
            Trace.WriteLine($"[{typeof(Kernel).Name}.{MethodBase.GetCurrentMethod().Name}] Disposed shader");
#endif
        }

        /// <summary>ファイナライザ</summary>
        ~Kernel() {
            Dispose();
        }
    }
}
