namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>1次元ストライドプール</summary>
    public sealed class StridePool1D : Pool1D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public StridePool1D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void stridepool_1d(float *inmap, float *outmap, 
                                          unsigned int outwidth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY};

                if (ch >= {Channels} || ox >= outwidth) {{
                    return;
                }}

                unsigned int ix = ox * {Stride};

                unsigned int inmap_idx = ch + {Channels} * ix;
                unsigned int outmap_idx = ch + {Channels} * ox;

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "stridepool_1d");
        }
    }
}
