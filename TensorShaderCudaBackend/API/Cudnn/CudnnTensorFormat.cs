using System;

namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {
        internal enum TensorFormat: Int32 {
            Nchw       = 0,
            Nhwc       = 1,
            NchwVectC  = 2,
        }
    }
}
