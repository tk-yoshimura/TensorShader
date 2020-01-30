namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>定数2項演算</summary>
    public sealed class BinaryConstantArithmetric : ConstantElementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(c + #x);</remarks>
        public BinaryConstantArithmetric(string name, string func)
            : base(constants: 1, arrays: 2, name) {
            string code = $@"
            __global__ void {name}(float c, const float* __restrict__ x, float* __restrict__ y, unsigned int length) {{
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
