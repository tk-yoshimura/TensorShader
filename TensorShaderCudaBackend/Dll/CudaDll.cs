using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace TensorShaderCudaBackend.Dll {
    static class CudaDll {
        public static NativeDll Cuda { get; private set; } = null;
        public static NativeDll Nvcuda { get; private set; } = null;
        public static NativeDll Nvrtc { get; private set; } = null;
        public static NativeDll Cudnn { get; private set; } = null;

        static CudaDll() {
            foreach (string libname in NvcudaLibraryNames) {
                if (NativeDll.Exists(libname, out NativeDll lib)) {
                    Nvcuda = lib;
                    break;
                }
            }
            if (Nvcuda is not null) {
                Trace.WriteLine($"[{typeof(CudaDll).Name}] {Nvcuda} loaded.");
            }

            foreach (string libname in CudaLibraryNames) {
                if (NativeDll.Exists(libname, out NativeDll lib)) {
                    Cuda = lib;
                    break;
                }
            }
            if (Cuda is not null) {
                Trace.WriteLine($"[{typeof(CudaDll).Name}] {Cuda} loaded.");
            }

            foreach (string libname in NvrtcLibraryNames) {
                if (NativeDll.Exists(libname, out NativeDll lib)) {
                    Nvrtc = lib;
                    break;
                }
            }
            if (Nvrtc is not null) {
                Trace.WriteLine($"[{typeof(CudaDll).Name}] {Nvrtc} loaded.");
            }

            if (Nvcuda is null || Cuda is null || Nvrtc is null) {
                throw new DllNotFoundException("Not found cuda library. (major version=10,11)");
            }

            foreach (string libname in CudnnLibraryNames) {
                if (NativeDll.Exists(libname, out NativeDll lib)) {
                    Cudnn = lib;
                    break;
                }
            }
            if (Cudnn is not null) {
                Trace.WriteLine($"[{typeof(CudaDll).Name}] {Cudnn} loaded.");
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

        static IEnumerable<string> CudnnLibraryNames {
            get {
                int[] cudnn_versions = new int[] { 8, 7 };

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                    foreach (int version in cudnn_versions) {
                        yield return $"cudnn64_{version}.dll";
                    }

                    yield return "cudnn64.dll";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                    foreach (int version in cudnn_versions) {
                        yield return $"libcudnn.so.{version}";
                    }

                    yield return "libcudnn.so";
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                    foreach (int version in cudnn_versions) {
                        yield return $"libcudnn.{version}.dylib";
                    }

                    yield return "cudnn.dylib";
                }
                else {
                    throw new NotSupportedException(RuntimeInformation.OSDescription);
                }
            }
        }
    }
}
