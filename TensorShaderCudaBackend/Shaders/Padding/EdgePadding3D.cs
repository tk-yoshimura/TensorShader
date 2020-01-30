namespace TensorShaderCudaBackend.Shaders.Padding {

    /// <summary>3次元エッジパディング</summary>
    public sealed class EdgePadding3D : Padding3D {

        /// <summary>コンストラクタ</summary>
        public EdgePadding3D(uint channels, uint pad_left, uint pad_right, uint pad_top, uint pad_bottom, uint pad_front, uint pad_rear)
            : base(channels, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear) {

            string code = $@"

            __global__ void edgepadding_3d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                          unsigned int oz,
                                          unsigned int inwidth, unsigned int outwidth,
                                          unsigned int inheight, unsigned int outheight,
                                          unsigned int indepth, unsigned int outdepth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                unsigned int ix = min(max(ox, {PadLeft}), {PadLeft} + inwidth - 1) - {PadLeft};
                unsigned int iy = min(max(oy, {PadTop}), {PadTop} + inheight - 1) - {PadTop};
                unsigned int iz = min(max(oz, {PadFront}), {PadFront} + indepth - 1) - {PadFront};
                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + inheight * iz));
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "edgepadding_3d");
        }
    }
}
