namespace TensorShaderCudaBackend.Shaders.Padding {

    /// <summary>2次元ゼロパディング</summary>
    public sealed class ZeroPadding2D : Padding2D {

        /// <summary>コンストラクタ</summary>
        public ZeroPadding2D(uint channels, uint pad_left, uint pad_right, uint pad_top, uint pad_bottom)
            : base(channels, pad_left, pad_right, pad_top, pad_bottom) {

            string code = $@"

            __global__ void zeropadding_2d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                          unsigned int inwidth, unsigned int outwidth,
                                          unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                unsigned int ix = ox - {PadLeft}, iy = oy - {PadTop};
                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * iy);
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * oy);

                float v = (ox >= {PadLeft} && ox < ({PadLeft} + inwidth)
                        && oy >= {PadTop} && oy < ({PadTop} + inheight))
                           ? inmap[inmap_idx] : 0.0;
                outmap[outmap_idx] = v;
            }}";

            this.Kernel = new Kernel(code, "zeropadding_2d");
        }
    }
}
