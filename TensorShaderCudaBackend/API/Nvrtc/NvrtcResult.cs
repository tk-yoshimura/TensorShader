using System;

namespace TensorShaderCudaBackend.API {
    public static partial class Nvrtc {
        internal enum ResultCode: Int32 {
            Success = 0,
            FailureOutOfMemory = 1,
            FailureProgramCreation = 2,
            FailureInvalidInput = 3,
            FailureInvalidProgram = 4,
            FailureInvalidOption = 5,
            FailureCompilation = 6,
            FailureBuiltinOperation = 7,
            FailureNoNameExpressionsAfterCompilation = 8,
            FailureNoLoweredNamesBeforeCompilation = 9,
            FailureNameExpressionNotValid = 10,
            FailureInternalError = 11
        }
    }
}
