namespace TensorShaderCudaBackend.Shaders.Zoom {

    /// <summary>最近傍補間</summary>
    public sealed class LinearZoom2D : Zoom2D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public LinearZoom2D(uint channels)
            : base(channels) {

            string code = $@"

            __global__ void linearzoom2d(float *inmap, float *outmap, 
                                         unsigned int inwidth, unsigned int outwidth, 
                                         unsigned int inheight) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY}, iy = {Defines.IndexZ};

                if (ch >= {Channels} || ix >= inwidth || iy >= inheight) {{
                    return;
                }}

                unsigned int ox = ix * 2, oy = iy * 2;
                unsigned int lu = 0, ru = {Channels}, ld = outwidth * {Channels}, rd = ru + ld;

                unsigned int ixl = ((ix > 0) ? ix - 1 : 0);
                unsigned int ixr = ((ix < inwidth - 1) ? ix + 1 : inwidth - 1);
                unsigned int iyu = ((iy > 0) ? iy - 1 : 0);
                unsigned int iyd = ((iy < inheight - 1) ? iy + 1 : inheight - 1);

                unsigned int inmap_c_idx  = ch + {Channels} * (ix  + inwidth * iy );
                unsigned int inmap_l_idx  = ch + {Channels} * (ixl + inwidth * iy );
                unsigned int inmap_r_idx  = ch + {Channels} * (ixr + inwidth * iy );
                unsigned int inmap_u_idx  = ch + {Channels} * (ix  + inwidth * iyu);
                unsigned int inmap_d_idx  = ch + {Channels} * (ix  + inwidth * iyd);
                unsigned int inmap_lu_idx = ch + {Channels} * (ixl + inwidth * iyu);
                unsigned int inmap_ru_idx = ch + {Channels} * (ixr + inwidth * iyu);
                unsigned int inmap_ld_idx = ch + {Channels} * (ixl + inwidth * iyd);
                unsigned int inmap_rd_idx = ch + {Channels} * (ixr + inwidth * iyd);

                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * oy);

                float xc  = inmap[inmap_c_idx] * 4;
                float xl  = inmap[inmap_l_idx] * 2;
                float xr  = inmap[inmap_r_idx] * 2;
                float xu  = inmap[inmap_u_idx] * 2;
                float xd  = inmap[inmap_d_idx] * 2;
                float xlu = inmap[inmap_lu_idx];
                float xru = inmap[inmap_ru_idx];
                float xld = inmap[inmap_ld_idx];
                float xrd = inmap[inmap_rd_idx];

                outmap[outmap_idx + lu] = (xc + xl + xu + xlu) / 9;
                outmap[outmap_idx + ru] = (xc + xr + xu + xru) / 9;
                outmap[outmap_idx + ld] = (xc + xl + xd + xld) / 9;
                outmap[outmap_idx + rd] = (xc + xr + xd + xrd) / 9;
            }}";

            this.Kernel = new Kernel(code, "linearzoom2d");
        }
    }
}
