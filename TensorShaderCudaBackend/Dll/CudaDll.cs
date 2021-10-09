using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.Dll {
    static class CudaDll {
        public static NativeDll Cuda { get; private set; } = null;
        public static NativeDll Nvcuda { get; private set; } = null;
        public static NativeDll Nvrtc { get; private set; } = null;

        static CudaDll() {
            foreach (string libname in NvcudaLibraryNames) {
                if (NativeDll.Exists(libname, out NativeDll lib)) {
                    Nvcuda = lib;
                    break;
                }
            }

            foreach (string libname in CudaLibraryNames) {
                if (NativeDll.Exists(libname, out NativeDll lib)) {
                    Cuda = lib;
                    break;
                }
            }

            foreach (string libname in NvrtcLibraryNames) {
                if (NativeDll.Exists(libname, out NativeDll lib)) {
                    Nvrtc = lib;
                    break;
                }
            }

            if (Nvcuda is null || Cuda is null || Nvrtc is null) {
                throw new DllNotFoundException("Not found cuda library. (major version=10,11)");
            }
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

        static IEnumerable<string> NvcudaLibraryNames {
            get {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                    yield return "nvcuda.dll";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                    yield return "libcuda.so";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                    yield return "libcuda.dylib";
                }
                else {
                    throw new NotSupportedException(RuntimeInformation.OSDescription);
                }
            }
        }

        static IEnumerable<string> CudaLibraryNames {
            get {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                    foreach ((int major, int minor) in VersionList) {
                        yield return $"cudart64_{major}{minor}.dll";
                    }

                    yield return "cudart64.dll";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                    foreach ((int major, int minor) in VersionList) {
                        yield return $"libcudart.so.{major}.{minor}";
                    }

                    yield return "libcudart.so";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                    foreach ((int major, int minor) in VersionList) {
                        yield return $"libcudart.{major}.{minor}.dylib";
                    }

                    yield return "libcudart.dylib";
                }
                else {
                    throw new NotSupportedException(RuntimeInformation.OSDescription);
                }
            }
        }

        static IEnumerable<string> NvrtcLibraryNames {
            get {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                    foreach ((int major, int minor) in VersionList) {
                        yield return $"nvrtc64_{major}{minor}_0.dll";
                    }

                    yield return "nvrtc64.dll";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                    foreach ((int major, int minor) in VersionList) {
                        yield return $"libnvrtc.so.{major}.{minor}";
                    }

                    yield return "libnvrtc.so";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                    foreach ((int major, int minor) in VersionList) {
                        yield return $"libnvrtc.{major}.{minor}.dylib";
                    }

                    yield return "libnvrtc.dylib";
                }
                else {
                    throw new NotSupportedException(RuntimeInformation.OSDescription);
                }
            }
        }
    }
}
