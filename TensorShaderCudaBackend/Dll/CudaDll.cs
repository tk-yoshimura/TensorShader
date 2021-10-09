using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.Dll {
    static class CudaDll {
        public static NativeDll Cuda { get; private set; }
        public static NativeDll Nvcuda { get; private set; }
        public static NativeDll Nvrtc { get; private set; }

        static CudaDll() {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                Nvcuda = new NativeDll("nvcuda.dll");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                Nvcuda = new NativeDll("libcuda.so");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                Nvcuda = new NativeDll("libcuda.dylib");
            }
            else { 
                throw new NotSupportedException(RuntimeInformation.OSDescription);
            }

            foreach ((string cuda, string nvrtc) in LibraryNames) {
                if (!NativeDll.Exists(cuda) || !NativeDll.Exists(nvrtc)) {
                    continue;
                }

                Cuda = new NativeDll(cuda);
                Nvrtc = new NativeDll(nvrtc);
                return;
            }

            throw new DllNotFoundException("Not found cuda library. (major version=10,11)");
        }

        static IEnumerable<(int major, int minor)> VersionList {
            get {
                for (int major = 11; major >= 10; major--) {
                    for (int minor = 9; minor >= 0; minor--) {
                        yield return (major, minor);
                    }
                }
            }
        }

        static IEnumerable<(string cuda, string nvrtc)> LibraryNames {
            get {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                    foreach ((int major, int minor) in VersionList) {
                        yield return ($"cudart64_{major}{minor}.dll", $"nvrtc64_{major}{minor}_0.dll");
                    }

                    yield return ("cudart64.dll", "nvrtc64.dll");
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                    foreach ((int major, int minor) in VersionList) {
                        yield return ($"libcudart.so.{major}.{minor}", $"libnvrtc.so.{major}.{minor}");
                    }

                    yield return ("libcudart.so", "libnvrtc.so");
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                    foreach ((int major, int minor) in VersionList) {
                        yield return ($"libcudart.{major}.{minor}.dylib", $"libnvrtc.{major}.{minor}.dylib");
                    }

                    yield return ("libcudart.dylib", "libnvrtc.dylib");
                }
                else {
                    throw new NotSupportedException(RuntimeInformation.OSDescription);
                }
            }
        }
    }
}
