namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>指数3項演算</summary>
    public sealed class TrinaryBiFactorArithmetric : FactorElementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(c1, c2, #x);</remarks>
        public TrinaryBiFactorArithmetric(string name, string func)
            : base(factors: 2, arrays: 2, name) {
            string code = $@"

            __constant__ float c1, c2;

            __global__ void {name}(const float* __restrict__ x, float* __restrict__ y, unsigned int length) {{
                unsigned int i = {Defines.IndexX};
                if (i >= length) {{
                    return;
                }}
                {func.Replace("#x", "x[i]").Replace("#y", "y[i]")}
            }}";

            this.Kernel = new Kernel(code, name);
        }
    }
}
