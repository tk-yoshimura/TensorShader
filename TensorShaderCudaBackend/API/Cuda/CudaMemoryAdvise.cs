using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorShaderCudaBackend.API {
    public static partial class Cuda {
#pragma warning disable IDE1006 // 命名スタイル
        /// <summary>CUDA memory advise types.</summary>
        internal enum cudaMemoryAdvise {
            SetReadMostly = 1,
            UnsetReadMostly = 2,
            SetPreferredLocation = 3,
            UnsetPreferredLocation = 4,
            SetAccessedBy = 5,
            UnsetAccessedBy = 6
        }
#pragma warning restore IDE1006 // 命名スタイル
    }
}
