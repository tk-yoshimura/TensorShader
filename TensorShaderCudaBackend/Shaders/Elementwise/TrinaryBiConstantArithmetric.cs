namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>定数3項演算</summary>
    public sealed class TrinaryBiConstantArithmetric : ConstantElementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(c1, c2, #x);</remarks>
        public TrinaryBiConstantArithmetric(string name, string func)
            : base(constants: 2, arrays: 2, name) {
            string code = $@"
            __global__ void {name}(float c1, float c2, const float* __restrict__ x, float* __restrict__ y, unsigned int length) {{
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
