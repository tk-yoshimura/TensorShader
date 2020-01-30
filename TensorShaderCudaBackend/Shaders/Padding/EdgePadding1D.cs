namespace TensorShaderCudaBackend.Shaders.Padding {

    /// <summary>1次元エッジパディング</summary>
    public sealed class EdgePadding1D : Padding1D {

        /// <summary>コンストラクタ</summary>
        public EdgePadding1D(uint channels, uint pad_left, uint pad_right)
            : base(channels, pad_left, pad_right) {

            string code = $@"

            __global__ void edgepadding_1d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                          unsigned int inwidth, unsigned int outwidth) {{

                unsigned int ch = {Defines.IndexX}, ox = {Defines.IndexY};

                if (ch >= {Channels} || ox >= outwidth) {{
                    return;
                }}

                unsigned int ix = min(max(ox, {PadLeft}), {PadLeft} + inwidth - 1) - {PadLeft};
                unsigned int inmap_idx = ch + {Channels} * ix;
                unsigned int outmap_idx = ch + {Channels} * ox;

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "edgepadding_1d");
        }
    }
}
