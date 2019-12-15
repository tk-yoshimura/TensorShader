namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>指数2項演算</summary>
    public sealed class BinaryFactorArithmetric : FactorElementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(c + #x);</remarks>
        public BinaryFactorArithmetric(string name, string func)
            : base(factors: 1, arrays: 2, name) {
            string code = $@"

            __constant__ float c;

            __global__ void {name}(float *x, float *y, unsigned int length) {{
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
