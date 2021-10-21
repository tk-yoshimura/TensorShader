namespace TensorShaderCudaBackend {
    /// <summary>環境情報</summary>
    public static class Environment {

        /// <summary>Cudnn libraryが存在するか</summary>
        public static bool CudnnExists => API.Cudnn.Exists;

        /// <summary>Cudnn libraryが有効か</summary>
        public static bool CudnnEnabled { set; get; } = CudnnExists;
    }
}
