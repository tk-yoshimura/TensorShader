namespace TensorShaderCudaBackend.Shaders.Padding {

    /// <summary>3次元ゼロパディング</summary>
    public sealed class ZeroPadding3D : Padding3D {

        /// <summary>コンストラクタ</summary>
        public ZeroPadding3D(uint channels, uint pad_left, uint pad_right, uint pad_top, uint pad_bottom, uint pad_front, uint pad_rear)
            : base(channels, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear) {

            string code = $@"

            __global__ void zeropadding_3d(float *inmap, float *outmap, 
                                          unsigned int oz,
                                          unsigned int inwidth, unsigned int outwidth, 
                                          unsigned int inheight, unsigned int outheight, 
                                          unsigned int indepth, unsigned int outdepth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY}, oy = {Defines.IndexZ};

                if (ch >= {Channels} || ox >= outwidth || oy >= outheight) {{
                    return;
                }}

                unsigned int ix = ox - {PadLeft}, iy = oy - {PadTop}, iz = oz - {PadFront};
                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + inheight * iz));
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));

                float v = (ox >= {PadLeft} && ox < ({PadLeft} + inwidth)
                        && oy >= {PadTop} && oy < ({PadTop} + inheight)
                        && oz >= {PadFront} && oz < ({PadFront} + indepth))
                           ? inmap[inmap_idx] : 0.0;
                outmap[outmap_idx] = v;
            }}";

            this.Kernel = new Kernel(code, "zeropadding_3d");
        }
    }
}
