namespace TensorShaderCudaBackend.API {
    public static partial class Nvcuda {
        internal enum FuncCache {
            PreferNone = 0x00,
            PreferShared = 0x01,
            PreferL1 = 0x02,
            PreferEqual = 0x03
        }
    }
}
