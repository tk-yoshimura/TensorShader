namespace TensorShaderCudaBackend.Shaders.Channelwise {

    /// <summary>チャネル参照2項演算</summary>
    public sealed class ChannelwiseBinaryArithmetric : Channelwise {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="name">関数名</param>
        /// <param name="func">関数</param>
        /// <remarks>func e.g. #y = f(#v, #x);</remarks>
        public ChannelwiseBinaryArithmetric(string name, string func, uint channels)
            : base(channels, map_arrays: 2, name) {
            string code;

            if (UseConstMemory) {
                code = $@"

                __constant__ float v[{Channels}];

                __global__ void {name}(const float* __restrict__ x, float* __restrict__ y, unsigned int length) {{
                    unsigned int i = {Defines.IndexX};
                    if (i >= length) {{
                        return;
                    }}
                    {func.Replace("#v", $"v[i % {Channels}]").Replace("#x", "x[i]").Replace("#y", "y[i]")};
                }}";
            }
            else {
                code = $@"

                __global__ void {name}(const float* __restrict__ v, const float* __restrict__ x, 
                                       float* __restrict__ y, unsigned int length) {{
                    unsigned int i = {Defines.IndexX};
                    if (i >= length) {{
                        return;
                    }}
                    {func.Replace("#v", $"v[i % {Channels}]").Replace("#x", "x[i]").Replace("#y", "y[i]")};
                }}";
            }

            this.Kernel = new Kernel(code, name);
        }
    }
}
