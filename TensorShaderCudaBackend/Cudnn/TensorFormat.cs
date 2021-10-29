using System;

namespace TensorShaderCudaBackend.Cudnn {
#pragma warning disable CS1591
    public enum TensorFormat : Int32 {
        NCHW = 0,
        NHWC = 1,
        NCHWVectC = 2,
    }
#pragma warning restore CS1591
}
