using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace TensorShaderCudaBackend.API {
    using size_t = Int64;

    public static partial class Cudnn {
        /// <summary>cudnnConvolutionFwdAlgoPerf_t</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ConvolutionFwdAlgoPerf {
            internal ConvolutionFwdAlgo algo;
            internal Status status;
            internal float time;
            internal size_t memory;
            internal Determinism determinism;
            internal MathType math_type;

            internal int reserved0, reserved1, reserved2;
        }

        /// <summary>cudnnConvolutionBwdDataAlgoPerf_t</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ConvolutionBwdDataAlgoPerf {
            internal ConvolutionBwdDataAlgo algo;
            internal Status status;
            internal float time;
            internal size_t memory;
            internal Determinism determinism;
            internal MathType math_type;

            internal int reserved0, reserved1, reserved2;
        }

        /// <summary>cudnnConvolutionBwdFilterAlgoPerf_t</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ConvolutionBwdFilterAlgoPerf {
            internal ConvolutionBwdFilterAlgo algo;
            internal Status status;
            internal float time;
            internal size_t memory;
            internal Determinism determinism;
            internal MathType math_type;

            internal int reserved0, reserved1, reserved2;
        }
    }
}
