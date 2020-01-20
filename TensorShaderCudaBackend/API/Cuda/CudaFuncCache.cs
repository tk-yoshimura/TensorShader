using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorShaderCudaBackend.API {
    public static partial class Cuda {

        /// <summary>共有メモリ/L1キャッシュ配分</summary>
        public enum FuncCache {
            /// <summary>デフォルト</summary>
            PreferNone   = 0,
            /// <summary>共有メモリ優先</summary>
            PreferShared = 1,
            /// <summary>L1キャッシュ優先</summary>
            PreferL1     = 2,
            /// <summary>等配分</summary>
            PreferEqual  = 3,
        }
    }
}
