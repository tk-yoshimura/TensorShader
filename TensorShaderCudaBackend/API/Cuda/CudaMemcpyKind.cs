namespace TensorShaderCudaBackend.API {
    public static partial class Cuda {
        /// <summary>CUDA memory copy types.</summary>
        private enum cudaMemcpyKind {
            HostToHost = 0,
            HostToDevice = 1,
            DeviceToHost = 2,
            DeviceToDevice = 3,
            Default = 4
        }
    }
}
