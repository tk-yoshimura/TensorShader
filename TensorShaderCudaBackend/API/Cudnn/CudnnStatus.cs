using System;

namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {
        internal enum Status : Int32 {
            Success = 0,
            NotInitialized = 1,
            AllocFailed = 2,
            BadParam = 3,
            InternalError = 4,
            InvalidValue = 5,
            ArchMismatch = 6,
            MappingError = 7,
            ExecutionFailed = 8,
            NotSupported = 9,
            LicenseError = 10,
            RuntimePrerequisiteMissing = 11,
            RuntimeInProgress = 12,
            RuntimeFpOverflow = 13
        }
    }
}
