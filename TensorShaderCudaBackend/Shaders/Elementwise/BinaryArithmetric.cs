namespace TensorShaderCudaBackend.Shaders.Elementwise {

    /// <summary>2項演算</summary>
    public sealed class BinaryArithmetric : Elementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(#x1, #x2);</remarks>
        public BinaryArithmetric(string name, string func)
            : base(arrays: 3, name) {
            string code = $@"
            __global__ void {name}(const float* __restrict__ x1, const float* __restrict__ x2, float* __restrict__ y, unsigned int length) {{
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
