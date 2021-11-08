namespace TensorShaderCudaBackend.API {
    public static partial class Cudnn {
        internal enum ConvolutionFwdPreference: int {
            NoWorkspace = 0,
            PreferFastest = 1,
            SpecifyWorkspaceLimit = 2,
        }

        internal enum ConvolutionFwdAlgo: int {
            ImplicitGemm = 0,
            ImplicitPrecompGemm = 1,
            Gemm = 2,
            Direct = 3,
            Fft = 4,
            FftTiling = 5,
            Winograd = 6,
            WinogradNonfused = 7,
            Count = 8
        }

        internal enum ConvolutionBwdFilterPreference: int {
            NoWorkspace = 0,
            PreferFastest = 1,
            SpecifyWorkspaceLimit = 2,
        }

        internal enum ConvolutionBwdFilterAlgo: int {
            Algo0 = 0,
            Algo1 = 1,
            Fft = 2,
            Algo3 = 3,
            Winograd = 4,
            WinogradNonfused = 5,
            FftTiling = 6,
            Count = 7
        }

        internal enum ConvolutionBwdDataPreference: int {
            NoWorkspace = 0,
            PreferFastest = 1,
            SpecifyWorkspaceLimit = 2,
        }

        internal enum ConvolutionBwdDataAlgo: int {
            Algo0 = 0,
            Algo1 = 1,
            Fft = 2,
            FftTiling = 3,
            Winograd = 4,
            WinogradNonfused = 5,
            Count = 6
        }

        internal enum ConvolutionMode: int {
            Convolution = 0,
            CrossCorrelation = 1
        }

        internal enum Determinism: int {
            NonDeterministic = 0,
            Deterministic = 1,
        }

        internal enum MathType: int {
            DefaultMath = 0,
            TensorOpMath = 1,
            TensorOpMathArrowConversion = 2,
            FmaMath = 3,
        }
    }
}
