namespace TensorShaderCudaBackend.Shaders.Zoom {

    /// <summary>最近傍補間</summary>
    public sealed class NeighborZoom1D : Zoom1D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        public NeighborZoom1D(uint channels)
            : base(channels) {

            string code = $@"

            __global__ void neighborzoom_1d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                            unsigned int inwidth) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY};

                if (ch >= {Channels} || ix >= inwidth) {{
                    return;
                }}

                unsigned int ox = ix * 2;
                unsigned int l = 0, r = {Channels};

                unsigned int inmap_idx = ch + {Channels} * ix;
                unsigned int outmap_idx = ch + {Channels} * ox;

                float x = inmap[inmap_idx];

                outmap[outmap_idx + l] = x;
                outmap[outmap_idx + r] = x;
            }}";

            this.Kernel = new Kernel(code, "neighborzoom_1d");
        }
    }
}
