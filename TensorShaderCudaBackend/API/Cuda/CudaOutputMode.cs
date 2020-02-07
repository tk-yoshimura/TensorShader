namespace TensorShaderCudaBackend.API {
    public static partial class Cuda {
#pragma warning disable IDE1006 // 命名スタイル
        private enum cudaOutputMode {
            cudaKeyValuePair = 0x00,
            cudaCSV = 0x01
        }
#pragma warning restore IDE1006 // 命名スタイル
    }
}
