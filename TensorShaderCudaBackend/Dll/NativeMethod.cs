using System;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.Dll {
    class NativeMethod<Delegate> where Delegate : System.Delegate {

        private readonly Delegate method;
        internal NativeDll Assembly { private set; get; }
        internal IntPtr Address { private set; get; } = IntPtr.Zero;
        public string Name { private set; get; } = string.Empty;

        public NativeMethod(NativeDll assembly, string funcname) {
            if (!assembly.IsValid) {
                throw new ArgumentException(
                    $"The specified assembly {assembly.Name} has not been loaded.",
                    nameof(assembly)
                );
            }

            this.Assembly = assembly;
            this.Address = NativeLibrary.GetExport(assembly.Handle, funcname);
            this.method = Marshal.GetDelegateForFunctionPointer<Delegate>(Address);
            this.Name = funcname;
        }

        public bool IsValid => Assembly.IsValid && Address != IntPtr.Zero;

        public static bool Exists(NativeDll assembly, string funcname) {
            if (!assembly.IsValid) {
                throw new ArgumentException(
                    $"The specified assembly {assembly.Name} has not been loaded.",
                    nameof(assembly)
                );
            }

            return NativeLibrary.TryGetExport(assembly.Handle, funcname, out IntPtr address) && (address != IntPtr.Zero);
        }

        public Delegate AsDelegate() {
            return IsValid ? method : throw new InvalidOperationException();
        }

        public override string ToString() {
            return IsValid ? $"{Assembly.Name} - {Name}" : "Missing entry point.";
        }
    }
}
