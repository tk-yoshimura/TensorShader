namespace TensorShaderCudaBackend.Shaders.Padding {

    /// <summary>1次元ゼロパディング</summary>
    public sealed class ZeroPadding1D : Padding1D {

        /// <summary>コンストラクタ</summary>
        public ZeroPadding1D(uint channels, uint pad_left, uint pad_right)
            : base(channels, pad_left, pad_right) {

            string code = $@"

            __global__ void zeropadding_1d(float *inmap, float *outmap, 
                                          unsigned int inwidth, unsigned int outwidth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY};

                if (ch >= {Channels} || ox >= outwidth) {{
                    return;
                }}

                unsigned int ix = ox - {PadLeft};
                unsigned int inmap_idx = ch + {Channels} * ix;
                unsigned int outmap_idx = ch + {Channels} * ox;

                float v = (ox >= {PadLeft} && ox < ({PadLeft} + inwidth))
                           ? inmap[inmap_idx] : 0.0;
                outmap[outmap_idx] = v;
            }}";

            this.Kernel = new Kernel(code, "zeropadding_1d");
        }
    }
}
