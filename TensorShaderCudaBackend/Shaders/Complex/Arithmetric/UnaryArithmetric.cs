namespace TensorShaderCudaBackend.Shaders.Complex.Arithmetric {

    /// <summary>複素1項演算</summary>
    public sealed class UnaryArithmetric : Elementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(#x);</remarks>
        public UnaryArithmetric(string name, string func)
            : base(arrays: 2, name) {
            string code = $@"
            {InlineFunctions}
            __global__ void {name}(float2 *inmap, float2 *outmap, unsigned int n) {{
                unsigned int i = {Defines.IndexX};
                if (i >= n) {{
                    return;
                }}
                float2 y, x = inmap[i];

                {func.Replace("#x", "x").Replace("#y", "y")}

                outmap[i] = y;
            }}";

            this.Kernel = new Kernel(code, name);
        }
    }
}
