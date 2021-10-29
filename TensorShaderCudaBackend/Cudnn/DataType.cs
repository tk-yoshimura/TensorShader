using System;

namespace TensorShaderCudaBackend.Cudnn {
#pragma warning disable CS1591
    public enum DataType : Int32 {
        Float = 0,
        Double = 1,
        Half = 2,
        Int8 = 3,
        Int32 = 4,
        Int8x4 = 5,
        Uint8 = 6,
        Uint8x4 = 7,
        Int8x32 = 8,
    }
#pragma warning restore CS1591
}
