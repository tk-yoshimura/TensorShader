﻿using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend {

    public sealed partial class Kernel {

        /// <summary>引数</summary>
        private sealed class Arguments : IDisposable {
            public IntPtr Ptr { get; private set; } = IntPtr.Zero;
            public List<GCHandle> HandleList { get; set; } = null;

            public Arguments(params object[] args) {
                if(args.OfType<CudaArrayBase>().Select((arr) => arr.DeviceID).Distinct().Count() > 1) { 
                    throw new ArgumentException("Several CUDA arrays with different device IDs were specified in the kernel execution request.");
                }

                int length = args.Length;

                List<GCHandle> handle_list = new List<GCHandle>();
                List<long> ptr_list = new List<long>();

                foreach (object obj in args) {
                    GCHandle handle;

                    if (obj is CudaArrayBase array) {
                        handle = GCHandle.Alloc(array.Ptr, GCHandleType.Pinned);
                    }
                    else if (obj is ValueType) {
                        handle = GCHandle.Alloc(obj, GCHandleType.Pinned);
                    }
                    else {
                        throw new ArgumentException(nameof(args));
                    }

                    handle_list.Add(handle);
                    ptr_list.Add(handle.AddrOfPinnedObject().ToInt64());
                }

                this.HandleList = handle_list;

                this.Ptr = Marshal.AllocHGlobal(IntPtr.Size * length);
                Marshal.Copy(ptr_list.ToArray(), 0, Ptr, length);
            }

            public void Dispose() {
                if (Ptr != IntPtr.Zero) {
                    Marshal.FreeHGlobal(Ptr);
                    Ptr = IntPtr.Zero;
                }
                if (HandleList != null) {
                    foreach (GCHandle handle in HandleList) {
                        handle.Free();
                    }
                    HandleList.Clear();
                    HandleList = null;
                }

                GC.SuppressFinalize(this);
            }

            ~Arguments() {
                Dispose();
            }
        }
    }
}
