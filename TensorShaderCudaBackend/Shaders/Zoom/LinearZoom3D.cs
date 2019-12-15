﻿namespace TensorShaderCudaBackend.Shaders.Zoom {

    /// <summary>線形補間</summary>
    public sealed class LinearZoom3D : Zoom3D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public LinearZoom3D(uint channels)
            : base(channels) {

            string code = $@"

            __global__ void linearzoom3d(float *inmap, float *outmap, 
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

                unsigned int ixl = ((ix > 0) ? ix - 1 : 0);
                unsigned int ixr = ((ix < inwidth - 1) ? ix + 1 : inwidth - 1);
                unsigned int iyu = ((iy > 0) ? iy - 1 : 0);
                unsigned int iyd = ((iy < inheight - 1) ? iy + 1 : inheight - 1);
                unsigned int izf = ((iz > 0) ? iz - 1 : 0);
                unsigned int izr = ((iz < indepth - 1) ? iz + 1 : indepth - 1);

                unsigned int inmap_c_idx    = ch + {Channels} * (ix  + inwidth * (iy  + inheight * iz ));
                unsigned int inmap_xl_idx   = ch + {Channels} * (ixl + inwidth * (iy  + inheight * iz ));
                unsigned int inmap_xr_idx   = ch + {Channels} * (ixr + inwidth * (iy  + inheight * iz ));
                unsigned int inmap_yu_idx   = ch + {Channels} * (ix  + inwidth * (iyu + inheight * iz ));
                unsigned int inmap_yd_idx   = ch + {Channels} * (ix  + inwidth * (iyd + inheight * iz ));
                unsigned int inmap_zf_idx   = ch + {Channels} * (ix  + inwidth * (iy  + inheight * izf));
                unsigned int inmap_zr_idx   = ch + {Channels} * (ix  + inwidth * (iy  + inheight * izr));
                unsigned int inmap_xlyu_idx = ch + {Channels} * (ixl + inwidth * (iyu + inheight * iz ));
                unsigned int inmap_xryu_idx = ch + {Channels} * (ixr + inwidth * (iyu + inheight * iz ));
                unsigned int inmap_xlyd_idx = ch + {Channels} * (ixl + inwidth * (iyd + inheight * iz ));
                unsigned int inmap_xryd_idx = ch + {Channels} * (ixr + inwidth * (iyd + inheight * iz ));
                unsigned int inmap_xlzf_idx = ch + {Channels} * (ixl + inwidth * (iy  + inheight * izf));
                unsigned int inmap_xrzf_idx = ch + {Channels} * (ixr + inwidth * (iy  + inheight * izf));
                unsigned int inmap_xlzr_idx = ch + {Channels} * (ixl + inwidth * (iy  + inheight * izr));
                unsigned int inmap_xrzr_idx = ch + {Channels} * (ixr + inwidth * (iy  + inheight * izr));
                unsigned int inmap_yuzf_idx = ch + {Channels} * (ix  + inwidth * (iyu + inheight * izf));
                unsigned int inmap_ydzf_idx = ch + {Channels} * (ix  + inwidth * (iyd + inheight * izf));
                unsigned int inmap_yuzr_idx = ch + {Channels} * (ix  + inwidth * (iyu + inheight * izr));
                unsigned int inmap_ydzr_idx = ch + {Channels} * (ix  + inwidth * (iyd + inheight * izr));
                unsigned int inmap_xlyuzf_idx = ch + {Channels} * (ixl + inwidth * (iyu + inheight * izf));
                unsigned int inmap_xryuzf_idx = ch + {Channels} * (ixr + inwidth * (iyu + inheight * izf));
                unsigned int inmap_xlydzf_idx = ch + {Channels} * (ixl + inwidth * (iyd + inheight * izf));
                unsigned int inmap_xrydzf_idx = ch + {Channels} * (ixr + inwidth * (iyd + inheight * izf));
                unsigned int inmap_xlyuzr_idx = ch + {Channels} * (ixl + inwidth * (iyu + inheight * izr));
                unsigned int inmap_xryuzr_idx = ch + {Channels} * (ixr + inwidth * (iyu + inheight * izr));
                unsigned int inmap_xlydzr_idx = ch + {Channels} * (ixl + inwidth * (iyd + inheight * izr));
                unsigned int inmap_xrydzr_idx = ch + {Channels} * (ixr + inwidth * (iyd + inheight * izr));

                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));

                float xc      = inmap[inmap_c_idx] * 8;
                float xxl     = inmap[inmap_xl_idx] * 4;
                float xxr     = inmap[inmap_xr_idx] * 4;
                float xyu     = inmap[inmap_yu_idx] * 4;
                float xyd     = inmap[inmap_yd_idx] * 4;
                float xzf     = inmap[inmap_zf_idx] * 4;
                float xzr     = inmap[inmap_zr_idx] * 4;
                float xxlyu   = inmap[inmap_xlyu_idx] * 2;
                float xxryu   = inmap[inmap_xryu_idx] * 2;
                float xxlyd   = inmap[inmap_xlyd_idx] * 2;
                float xxryd   = inmap[inmap_xryd_idx] * 2;
                float xxlzf   = inmap[inmap_xlzf_idx] * 2;
                float xxrzf   = inmap[inmap_xrzf_idx] * 2;
                float xxlzr   = inmap[inmap_xlzr_idx] * 2;
                float xxrzr   = inmap[inmap_xrzr_idx] * 2;
                float xyuzf   = inmap[inmap_yuzf_idx] * 2;
                float xydzf   = inmap[inmap_ydzf_idx] * 2;
                float xyuzr   = inmap[inmap_yuzr_idx] * 2;
                float xydzr   = inmap[inmap_ydzr_idx] * 2;
                float xxlyuzf = inmap[inmap_xlyuzf_idx];
                float xxryuzf = inmap[inmap_xryuzf_idx];
                float xxlydzf = inmap[inmap_xlydzf_idx];
                float xxrydzf = inmap[inmap_xrydzf_idx];
                float xxlyuzr = inmap[inmap_xlyuzr_idx];
                float xxryuzr = inmap[inmap_xryuzr_idx];
                float xxlydzr = inmap[inmap_xlydzr_idx];
                float xxrydzr = inmap[inmap_xrydzr_idx];

                outmap[outmap_idx + luf] = (xc + xxl + xyu + xzf + xxlyu + xxlzf + xyuzf + xxlyuzf) / 27;
                outmap[outmap_idx + ruf] = (xc + xxr + xyu + xzf + xxryu + xxrzf + xyuzf + xxryuzf) / 27;
                outmap[outmap_idx + ldf] = (xc + xxl + xyd + xzf + xxlyd + xxlzf + xydzf + xxlydzf) / 27;
                outmap[outmap_idx + rdf] = (xc + xxr + xyd + xzf + xxryd + xxrzf + xydzf + xxrydzf) / 27;
                outmap[outmap_idx + lur] = (xc + xxl + xyu + xzr + xxlyu + xxlzr + xyuzr + xxlyuzr) / 27;
                outmap[outmap_idx + rur] = (xc + xxr + xyu + xzr + xxryu + xxrzr + xyuzr + xxryuzr) / 27;
                outmap[outmap_idx + ldr] = (xc + xxl + xyd + xzr + xxlyd + xxlzr + xydzr + xxlydzr) / 27;
                outmap[outmap_idx + rdr] = (xc + xxr + xyd + xzr + xxryd + xxrzr + xydzr + xxrydzr) / 27;
            }}";

            this.Kernel = new Kernel(code, "linearzoom3d");
        }
    }
}
