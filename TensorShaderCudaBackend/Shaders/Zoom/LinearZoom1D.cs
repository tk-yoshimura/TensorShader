namespace TensorShaderCudaBackend.Shaders.Zoom {

    /// <summary>線形補間</summary>
    public sealed class LinearZoom1D : Zoom1D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public LinearZoom1D(uint channels)
            : base(channels) {

            string code = $@"

            __global__ void linearzoom_1d(float *inmap, float *outmap,
                                         unsigned int inwidth) {{

                const float inv = 1.0 / 3;

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY};

                if (ch >= {Channels} || ix >= inwidth) {{
                    return;
                }}

                unsigned int ox = ix * 2;
                unsigned int l = 0, r = {Channels};

                unsigned int ixl = max(1, ix) - 1;
                unsigned int ixr = min(inwidth - 1, ix + 1);

                unsigned int inmap_c_idx = ch + {Channels} * ix;
                unsigned int inmap_l_idx = ch + {Channels} * ixl;
                unsigned int inmap_r_idx = ch + {Channels} * ixr;

                unsigned int outmap_idx = ch + {Channels} * ox;

                float xc = ldexpf(inmap[inmap_c_idx], 1);
                float xl = inmap[inmap_l_idx];
                float xr = inmap[inmap_r_idx];

                outmap[outmap_idx + l] = (xc + xl) * inv;
                outmap[outmap_idx + r] = (xc + xr) * inv;
            }}";

            this.Kernel = new Kernel(code, "linearzoom_1d");
        }
    }
}
