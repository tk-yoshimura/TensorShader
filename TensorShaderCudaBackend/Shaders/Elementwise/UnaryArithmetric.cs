namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>単項演算</summary>
    public sealed class UnaryArithmetric : Elementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(#x);</remarks>
        public UnaryArithmetric(string name, string func)
            : base(arrays: 2, name) {
            string code = $@"
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
