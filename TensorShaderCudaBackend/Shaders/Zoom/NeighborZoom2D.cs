namespace TensorShaderCudaBackend.Shaders.Zoom {

    /// <summary>最近傍補間</summary>
    public sealed class NeighborZoom2D : Zoom2D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public NeighborZoom2D(uint channels)
            : base(channels) {

            string code = $@"

            __global__ void neighborzoom_2d(float *inmap, float *outmap,
                                           unsigned int inwidth, unsigned int outwidth,
                                           unsigned int inheight) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY}, iy = {Defines.IndexZ};

                if (ch >= {Channels} || ix >= inwidth || iy >= inheight) {{
                    return;
                }}

                unsigned int ox = ix * 2, oy = iy * 2;
                unsigned int lu = 0, ru = {Channels}, ld = outwidth * {Channels}, rd = ru + ld;

                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * iy);
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * oy);

                float x = inmap[inmap_idx];

                outmap[outmap_idx + lu] = x;
                outmap[outmap_idx + ru] = x;
                outmap[outmap_idx + ld] = x;
                outmap[outmap_idx + rd] = x;
            }}";

            this.Kernel = new Kernel(code, "neighborzoom_2d");
        }
    }
}
