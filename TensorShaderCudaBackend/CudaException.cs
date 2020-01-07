using System;

using TensorShaderCudaBackend.API;

namespace TensorShaderCudaBackend {

    /// <summary>CUDA API 例外</summary>
    [Serializable]
    public class CudaException : Exception {

        internal CudaException(string str)
            : base(str) { }

        internal CudaException(Cuda.ErrorCode code)
            : base(Cuda.GetErrorString(code)) { }

        internal CudaException(Nvcuda.ResultCode code)
            : base(Nvcuda.GetErrorString(code)) { }

        internal CudaException(Nvrtc.ResultCode code)
            : base(Nvrtc.GetErrorString(code)) { }
    }
}
