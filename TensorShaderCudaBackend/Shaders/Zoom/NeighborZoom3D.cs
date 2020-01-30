namespace TensorShaderCudaBackend.Shaders.Zoom {

    /// <summary>最近傍補間</summary>
    public sealed class NeighborZoom3D : Zoom3D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public NeighborZoom3D(uint channels)
            : base(channels) {

            string code = $@"

            __global__ void neighborzoom_3d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                            unsigned int iz,
                                            unsigned int inwidth, unsigned int outwidth,
                                            unsigned int inheight, unsigned int outheight,
                                            unsigned int indepth) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY}, iy = {Defines.IndexZ};

                if (ch >= {Channels} || ix >= inwidth || iy >= inheight) {{
                    return;
                }}

                unsigned int ox = ix * 2, oy = iy * 2, oz = iz * 2;
                unsigned int luf = 0, ruf = {Channels}, ldf = outwidth * {Channels}, rdf = ruf + ldf;
                unsigned int lur = ldf * outheight, rur = lur + ruf, ldr = lur + ldf, rdr = lur + rdf;

                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + inheight * iz));
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));

                float x = inmap[inmap_idx];

                outmap[outmap_idx + luf] = x;
                outmap[outmap_idx + ruf] = x;
                outmap[outmap_idx + ldf] = x;
                outmap[outmap_idx + rdf] = x;
                outmap[outmap_idx + lur] = x;
                outmap[outmap_idx + rur] = x;
                outmap[outmap_idx + ldr] = x;
                outmap[outmap_idx + rdr] = x;
            }}";

            this.Kernel = new Kernel(code, "neighborzoom_3d");
        }
    }
}
