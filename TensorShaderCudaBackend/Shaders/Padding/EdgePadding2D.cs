namespace TensorShaderCudaBackend.Shaders.Padding {

    /// <summary>2次元エッジパディング</summary>
    public sealed class EdgePadding2D : Padding2D {

        /// <summary>コンストラクタ</summary>
        public EdgePadding2D(uint channels, uint pad_left, uint pad_right, uint pad_top, uint pad_bottom)
            : base(channels, pad_left, pad_right, pad_top, pad_bottom) {

            string code = $@"

            __global__ void edgepadding2d(float *inmap, float *outmap, 
                                          unsigned int inwidth, unsigned int outwidth, 
                                          unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                unsigned int ix = min(max(ox, {PadLeft}), {PadLeft} + inwidth - 1) - {PadLeft};
                unsigned int iy = min(max(oy, {PadTop}), {PadTop} + inheight - 1) - {PadTop};
                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * iy);
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * oy);

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "edgepadding2d");
        }
    }
}
