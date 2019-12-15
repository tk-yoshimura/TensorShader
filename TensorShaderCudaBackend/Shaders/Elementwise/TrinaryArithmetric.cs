namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>3項演算</summary>
    public sealed class TrinaryArithmetric : Elementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(#x1, #x2, #x3);</remarks>
        public TrinaryArithmetric(string name, string func)
            : base(arrays: 4, name) {
            string code = $@"
            __global__ void {name}(float *x1, float *x2, float *x3, float *y, unsigned int length) {{
                unsigned int i = {Defines.IndexX};
                if (i >= length) {{
                    return;
                }}
                {func.Replace("#x1", "x1[i]").Replace("#x2", "x2[i]").Replace("#x3", "x3[i]").Replace("#y", "y[i]")}
            }}";

            this.Kernel = new Kernel(code, name);
        }
    }
}
