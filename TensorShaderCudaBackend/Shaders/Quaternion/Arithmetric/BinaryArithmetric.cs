namespace TensorShaderCudaBackend.Shaders.Quaternion.Arithmetric {

    /// <summary>四元数2項演算</summary>
    public sealed class BinaryArithmetric : Elementwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(#x1, #x2);</remarks>
        public BinaryArithmetric(string name, string func)
            : base(arrays: 3, name) {
            string code = $@"

            {Defines.CtorFloat4}

            __global__ void {name}(float4 *inmap1, float4 *inmap2, float4 *outmap, unsigned int n) {{
                unsigned int i = {Defines.IndexX};
                if (i >= n) {{
                    return;
                }}
                float4 y, x1 = inmap1[i], x2 = inmap2[i];

                {func.Replace("#x1", "x1").Replace("#x2", "x2").Replace("#y", "y")}

                outmap[i] = y;
            }}";

            this.Kernel = new Kernel(code, name);
        }
    }
}
