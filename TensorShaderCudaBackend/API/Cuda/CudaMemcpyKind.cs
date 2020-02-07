namespace TensorShaderCudaBackend.API {
    public static partial class Cuda {
#pragma warning disable IDE1006 // 命名スタイル
        /// <summary>CUDA memory copy types.</summary>
        private enum cudaMemcpyKind {
            HostToHost = 0,
            HostToDevice = 1,
            DeviceToHost = 2,
            DeviceToDevice = 3,
            Default = 4
        }
#pragma warning restore IDE1006 // 命名スタイル
    }
}
