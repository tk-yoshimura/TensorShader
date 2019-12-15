namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>指数3項演算</summary>
    public sealed class TrinaryUniFactorArithmetric : FactorElementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(c, #x1, #x2);</remarks>
        public TrinaryUniFactorArithmetric(string name, string func)
            : base(factors : 1, arrays : 3, name) {
            string code = $@"

            __constant__ float c;

            __global__ void {name}(float *x1, float *x2, float *y, unsigned int length) {{
                unsigned int i = {Defines.IndexX};
                if (i >= length) {{
                    return;
                }}
                {func.Replace("#x1", "x1[i]").Replace("#x2", "x2[i]").Replace("#y", "y[i]")}
            }}";

            this.Kernel = new Kernel(code, name);
        }
    }
}
