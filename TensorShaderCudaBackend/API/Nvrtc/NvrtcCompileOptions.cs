namespace TensorShaderCudaBackend.API {
    public static partial class Nvrtc {

        /// <summary>コンパイルオプション</summary>
        internal static class CompileOptions {

            /// <summary>GPUアーキテクチャ</summary>
            /// <remarks>現在選択中のアーキテクチャが指定される</remarks>
            public static string ArchitectureTarget {
                get {
                    (int major, int minor) = (Cuda.CurrectDeviceProperty.Major, Cuda.CurrectDeviceProperty.Minor);

                    return $"--gpu-architecture=compute_{major}{minor}";
                }
            }

            /// <summary>デバッグ情報生成</summary>
            public static string Debug => "--device-debug";

        }

    }
}
