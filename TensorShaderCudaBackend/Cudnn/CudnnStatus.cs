using System;

namespace TensorShaderCudaBackend.Cudnn {
#pragma warning disable CS1591
    public enum Status : Int32 {
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
        RuntimeFpOverflow = 13,
        VersionMismatch = 14,
    }
#pragma warning restore CS1591
}
