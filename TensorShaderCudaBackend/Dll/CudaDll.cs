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

        public static (NativeDll ops_infer, NativeDll ops_train, NativeDll cnn_infer, NativeDll cnn_train) CudnnSubset { get; private set; } = (null, null, null, null);

        public static int CudnnVersion { get; private set; } = 0;

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

            foreach ((int version, string libname) in CudnnLibraryNames) {
                if (NativeDll.Exists(libname, out NativeDll lib)) {
                    Cudnn = lib;
                    CudnnVersion = version;
                    break;
                }
            }
            if (Cudnn is not null) {
                if (CudnnVersion >= 8) {
                    (string ops_infer, string ops_train, string cnn_infer, string cnn_train) = CudnnSubsetLibraryNames(CudnnVersion);

                    if (NativeDll.Exists(ops_infer, out NativeDll ops_infer_lib) && NativeDll.Exists(ops_train, out NativeDll ops_train_lib) &&
                        NativeDll.Exists(cnn_infer, out NativeDll cnn_infer_lib) && NativeDll.Exists(cnn_train, out NativeDll cnn_train_lib)) {

                        CudnnSubset = (ops_infer_lib, ops_train_lib, cnn_infer_lib, cnn_train_lib);
                    }
                    else {
                        throw new DllNotFoundException("Not found cudnn subset library. (major version=8)");
                    }
                }

                Trace.WriteLine($"[{typeof(CudaDll).Name}] {Cudnn} loaded.");
            }
        }

        static IReadOnlyList<(int major, int minor)> VersionList {
            get {
                List<(int major, int minor)> versions = new();

                for (int major = 11; major >= 10; major--) {
                    for (int minor = 9; minor >= 0; minor--) {
                        versions.Add((major, minor));
                    }
                }

                return versions;
            }
        }

        static IReadOnlyList<string> NvcudaLibraryNames {
            get {
                List<string> libnames = new();

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                    libnames.Add("nvcuda.dll");
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                    libnames.Add("libcuda.so");
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                    libnames.Add("libcuda.dylib");
                }
                else {
                    throw new NotSupportedException(RuntimeInformation.OSDescription);
                }

                return libnames;
            }
        }

        static IReadOnlyList<string> CudaLibraryNames {
            get {
                List<string> libnames = new();

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                    foreach ((int major, int minor) in VersionList) {
                        libnames.Add($"cudart64_{major}{minor}.dll");
                    }

                    libnames.Add("cudart64.dll");
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                    foreach ((int major, int minor) in VersionList) {
                        libnames.Add($"libcudart.so.{major}.{minor}");
                    }

                    libnames.Add("libcudart.so");
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                    foreach ((int major, int minor) in VersionList) {
                        libnames.Add($"libcudart.{major}.{minor}.dylib");
                    }

                    libnames.Add("libcudart.dylib");
                }
                else {
                    throw new NotSupportedException(RuntimeInformation.OSDescription);
                }

                return libnames;
            }
        }

        static IReadOnlyList<string> NvrtcLibraryNames {
            get {
                List<string> libnames = new();

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                    foreach ((int major, int minor) in VersionList) {
                        libnames.Add($"nvrtc64_{major}{minor}_0.dll");
                    }

                    libnames.Add("nvrtc64.dll");
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                    foreach ((int major, int minor) in VersionList) {
                        libnames.Add($"libnvrtc.so.{major}.{minor}");
                    }

                    libnames.Add("libnvrtc.so");
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                    foreach ((int major, int minor) in VersionList) {
                        libnames.Add($"libnvrtc.{major}.{minor}.dylib");
                    }

                    libnames.Add("libnvrtc.dylib");
                }
                else {
                    throw new NotSupportedException(RuntimeInformation.OSDescription);
                }

                return libnames;
            }
        }

        static IReadOnlyList<(int version, string libname)> CudnnLibraryNames {
            get {
                List<(int version, string libname)> libnames = new();

                int[] cudnn_versions = new int[] { 8, 7 };

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                    foreach (int version in cudnn_versions) {
                        libnames.Add((version, $"cudnn64_{version}.dll"));
                    }
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                    foreach (int version in cudnn_versions) {
                        libnames.Add((version, $"libcudnn.so.{version}"));
                    }
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                    foreach (int version in cudnn_versions) {
                        libnames.Add((version, $"libcudnn.{version}.dylib"));
                    }
                }
                else {
                    throw new NotSupportedException(RuntimeInformation.OSDescription);
                }

                return libnames;
            }
        }

        static (string ops_infer, string ops_train, string cnn_infer, string cnn_train) CudnnSubsetLibraryNames(int version) {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) {
                return ($"cudnn_ops_infer64_{version}.dll", $"cudnn_ops_train64_{version}.dll",
                        $"cudnn_cnn_infer64_{version}.dll", $"cudnn_cnn_train64_{version}.dll");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) {
                return ($"libcudnn_ops_infer.so.{version}", $"libcudnn_ops_train.so.{version}",
                        $"libcudnn_cnn_infer.so.{version}", $"libcudnn_cnn_train.so.{version}");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) {
                return ($"libcudnn_ops_infer.{version}.dylib", $"libcudnn_ops_train.{version}.dylib",
                        $"libcudnn_cnn_infer.{version}.dylib", $"libcudnn_cnn_train.{version}.dylib");
            }
            else {
                throw new NotSupportedException(RuntimeInformation.OSDescription);
            }
        }
    }
}
