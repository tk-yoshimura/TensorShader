namespace TensorShaderCudaBackend.Cudnn {
#pragma warning disable CS1591 // 公開されている型またはメンバーの XML コメントがありません
    public enum ConvolutionFwdAlgo : int {
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

    public enum ConvolutionBwdFilterAlgo : int {
        Algo0 = 0,
        Algo1 = 1,
        Fft = 2,
        Algo3 = 3,
        Winograd = 4,
        WinogradNonfused = 5,
        FftTiling = 6,
        Count = 7
    }

    public enum ConvolutionBwdDataAlgo : int {
        Algo0 = 0,
        Algo1 = 1,
        Fft = 2,
        FftTiling = 3,
        Winograd = 4,
        WinogradNonfused = 5,
        Count = 6
    }

    public enum ConvolutionMode : int {
        Convolution = 0,
        CrossCorrelation = 1
    }

    public enum Determinism : int {
        NonDeterministic = 0,
        Deterministic = 1,
    }

    public enum MathType : int {
        DefaultMath = 0,
        TensorOpMath = 1,
        TensorOpMathArrowConversion = 2,
        FmaMath = 3,
    }
#pragma warning restore CS1591 // 公開されている型またはメンバーの XML コメントがありません
}
