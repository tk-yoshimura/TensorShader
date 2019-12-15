using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

using TensorShaderCudaBackend.API;

namespace TensorShaderCudaBackend {

    /// <summary>Cuda配列基底クラス</summary>
    public abstract class CudaArrayBase {
        internal virtual IntPtr Ptr { get; }

        /// <summary>配列長</summary>
        public virtual ulong Length { protected set; get; }

        /// <summary>バイトサイズ</summary>
        public virtual ulong ByteSize { get; }

        /// <summary>最大バイトサイズ</summary>
        public static ulong MaxByteSize => 0x40000000ul;

        /// <summary>デバイスID</summary>
        public virtual int DeviceID { protected set; get; }

        /// <summary>有効か</summary>
        public virtual bool IsValid { get; }
    }

    /// <summary>Cuda配列</summary>
    public sealed class CudaArray<T> : CudaArrayBase, IDisposable where T : struct, IComparable {
        private IntPtr ptr;

        internal override IntPtr Ptr {
            get {
                if (ptr == IntPtr.Zero) {
                    throw new ObjectDisposedException(GetType().FullName);
                }

                return ptr;
            }
        }

        internal IntPtr ElementPtr(uint index) => (IntPtr)((ulong)Ptr.ToInt64() + index * ElementSize);

        /// <summary>配列長</summary>
        public override ulong Length { protected set; get; }

        /// <summary>最大配列長</summary>
        public static ulong MaxLength => MaxByteSize / ElementSize;

        /// <summary>バイトサイズ</summary>
        public override ulong ByteSize => ElementSize * Length;

        /// <summary>要素サイズ</summary>
        public static ulong ElementSize => (ulong)Marshal.SizeOf(typeof(T));

        /// <summary>有効か</summary>
        public override bool IsValid => ptr != IntPtr.Zero;

        /// <summary>要素型</summary>
        public static Type ElementType => typeof(T);

        /// <summary>値</summary>
        public T[] Value => this;

        /// <summary>概要</summary>
        public string Overview { private set; get; }

        /// <summary>コンストラクタ</summary>
        public CudaArray(ulong length, bool zeroset = true) {
            if (length > MaxLength) {
                throw new ArgumentOutOfRangeException(nameof(length));
            }

            this.ptr = Cuda.Memory.Allocate<T>(length);
            this.Length = length;
            this.Overview = $"{ElementType.Name}[{length}] @{Cuda.CurrectDeviceProperty.Name} ({Cuda.CurrectDeviceID})";
            this.DeviceID = Cuda.CurrectDeviceID;

            if (zeroset) {
                Zeroset();
            }

            if(DateTime.Now.Millisecond % 100 == 0) { 
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        /// <summary>コンストラクタ</summary>
        public CudaArray(T[] array)
            : this((ulong)array.LongLength, zeroset: false) {
            
            Write(array);
        }

        /// <summary>マネージ配列へ型変換</summary>
        public static implicit operator T[](CudaArray<T> array) {
            T[] arr = new T[array.Length];
            array.Read(arr);

            return arr;
        }

        /// <summary>マネージ配列から型変換</summary>
        public static implicit operator CudaArray<T>(T[] array) {
            return new CudaArray<T>(array);
        }

        /// <summary>GPUメモリへ書き込み</summary>
        public void Write(T[] array) {
            Cuda.Memory.CopyHostToDevice(array, ptr, Length);
        }

        /// <summary>GPUメモリから読み込み</summary>
        public void Read(T[] array) {
            Cuda.Memory.CopyDeviceToHost(ptr, array, Length);
        }

        /// <summary>GPUメモリへ書き込み</summary>
        public void Write(T[] array, ulong count) {
            Cuda.Memory.CopyHostToDevice(array, ptr, count);
        }

        /// <summary>GPUメモリから読み込み</summary>
        public void Read(T[] array, ulong count) {
            Cuda.Memory.CopyDeviceToHost(ptr, array, count);
        }

        /// <summary>ゼロクリア</summary>
        public void Zeroset() {
            Cuda.Memory.Zeroset<T>(ptr, Length);
        }

        /// <summary>ゼロクリア</summary>
        public void Zeroset(ulong count) {
            if (count > Length) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            Cuda.Memory.Zeroset<T>(ptr, count);
        }

        /// <summary>ゼロクリア</summary>
        public void Zeroset(ulong index, ulong count) {
            if (index >= Length || index + count > Length) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            IntPtr ref_ptr = (IntPtr)(ptr.ToInt64() + (long)(ElementSize * index));

            Cuda.Memory.Zeroset<T>(ref_ptr, count);
        }

        /// <summary>ゼロクリア</summary>
        public void ZerosetAsync(Stream stream) {
            Cuda.Memory.ZerosetAsync<T>(ptr, Length, stream.Ptr);
        }

        /// <summary>ゼロクリア</summary>
        public void ZerosetAsync(Stream stream, ulong count) {
            if (count > Length) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            Cuda.Memory.ZerosetAsync<T>(ptr, count, stream.Ptr);
        }

        /// <summary>ゼロクリア</summary>
        public void ZerosetAsync(Stream stream, ulong index, ulong count) {
            if (index >= Length || index + count > Length) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            IntPtr ref_ptr = (IntPtr)(ptr.ToInt64() + (long)(ElementSize * index));

            Cuda.Memory.ZerosetAsync<T>(ref_ptr, count, stream.Ptr);
        }

        /// <summary>GPUメモリ領域コピー</summary>
        public void CopyTo(CudaArray<T> array, ulong count) {
            Copy(this, array, count);
        }

        /// <summary>GPUメモリ領域コピー</summary>
        public void CopyTo(ulong index, CudaArray<T> dst_array, ulong dst_index, ulong count) {
            Copy(this, index, dst_array, dst_index, count);
        }

        /// <summary>GPUメモリ領域コピー</summary>
        public static void Copy(CudaArray<T> src_array, CudaArray<T> dst_array, ulong count) {
            if (count > src_array.Length || count > dst_array.Length) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            Cuda.Memory.CopyDeviceToDevice<T>(src_array.ptr, dst_array.ptr, count);
        }

        /// <summary>GPUメモリ領域コピー</summary>
        public static void Copy(CudaArray<T> src_array, ulong src_index, CudaArray<T> dst_array, ulong dst_index, ulong count) {
            if (src_index >= src_array.Length || src_index + count > src_array.Length) {
                throw new ArgumentOutOfRangeException(nameof(src_index));
            }
            if (dst_index >= dst_array.Length || dst_index + count > dst_array.Length) {
                throw new ArgumentOutOfRangeException(nameof(dst_index));
            }

            IntPtr src_ptr = (IntPtr)(src_array.ptr.ToInt64() + (long)(ElementSize * src_index));
            IntPtr dst_ptr = (IntPtr)(dst_array.ptr.ToInt64() + (long)(ElementSize * dst_index));

            Cuda.Memory.CopyDeviceToDevice<T>(src_ptr, dst_ptr, count);
        }

        /// <summary>GPUメモリ領域コピー</summary>
        public void CopyToAsync(Stream stream, CudaArray<T> array, ulong count) {
            CopyAsync(stream, this, array, count);
        }

        /// <summary>GPUメモリ領域コピー</summary>
        public void CopyToAsync(Stream stream, ulong index, CudaArray<T> dst_array, ulong dst_index, ulong count) {
            CopyAsync(stream, this, index, dst_array, dst_index, count);
        }

        /// <summary>GPUメモリ領域コピー</summary>
        public static void CopyAsync(Stream stream, CudaArray<T> src_array, CudaArray<T> dst_array, ulong count) {
            if (count > src_array.Length || count > dst_array.Length) {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            Cuda.Memory.CopyDeviceToDeviceAsync<T>(src_array.ptr, dst_array.ptr, count, stream.Ptr);
        }

        /// <summary>GPUメモリ領域コピー</summary>
        public static void CopyAsync(Stream stream, CudaArray<T> src_array, ulong src_index, CudaArray<T> dst_array, ulong dst_index, ulong count) {
            if (src_index >= src_array.Length || src_index + count > src_array.Length) {
                throw new ArgumentOutOfRangeException(nameof(src_index));
            }
            if (dst_index >= dst_array.Length || dst_index + count > dst_array.Length) {
                throw new ArgumentOutOfRangeException(nameof(dst_index));
            }

            IntPtr src_ptr = (IntPtr)(src_array.ptr.ToInt64() + (long)(ElementSize * src_index));
            IntPtr dst_ptr = (IntPtr)(dst_array.ptr.ToInt64() + (long)(ElementSize * dst_index));

            Cuda.Memory.CopyDeviceToDeviceAsync<T>(src_ptr, dst_ptr, count, stream.Ptr);
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return $"{nameof(CudaArray<T>)} {Overview}";
        }

        /// <summary>破棄</summary>
        public void Dispose() {
            Cuda.Memory.Free(ref ptr);
            Length = 0;
            Overview = null;
            DeviceID = 0;

            GC.SuppressFinalize(this);

#if DEBUG
            Trace.WriteLine("Disposed Gpu Array");
#endif
        }

        /// <summary>ファイナライザ</summary>
        ~CudaArray() {
            Dispose();
        }
    }
}
