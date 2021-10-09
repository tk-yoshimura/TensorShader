
using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.Dll {
    class NativeDll : IDisposable {
        internal IntPtr Handle { private set; get; } = IntPtr.Zero;

        public string Name { private set; get; } = string.Empty;

        public NativeDll(string libname) {
            this.Handle = NativeLibrary.Load(libname);
            this.Name = libname;
        }

        private NativeDll(IntPtr handle, string libname) {
            if (handle == IntPtr.Zero) {
                throw new ArgumentNullException(nameof(handle));
            }

            this.Handle = handle;
            this.Name = libname;
        }

        public bool IsValid => Handle != IntPtr.Zero;

        public static bool Exists(string libname) {
            if (!NativeLibrary.TryLoad(libname, out IntPtr handle) || (handle == IntPtr.Zero)) {
                return false;
            }

            NativeLibrary.Free(handle);

            return true;
        }

        public static bool Exists(string libname, out NativeDll dll) {
            dll = null;

            if (!NativeLibrary.TryLoad(libname, out IntPtr handle) || (handle == IntPtr.Zero)) {
                return false;
            }

            dll = new NativeDll(handle, libname);

            return true;
        }

        public override string ToString() {
            return IsValid ? Name : "Failed to load.";
        }

        public void Dispose() {
            if (Handle != IntPtr.Zero) {
                NativeLibrary.Free(Handle);
                Handle = IntPtr.Zero;
                Name = string.Empty;
            }

            GC.SuppressFinalize(this);
        }

        ~NativeDll() {
            Dispose();
        }
    }
}
