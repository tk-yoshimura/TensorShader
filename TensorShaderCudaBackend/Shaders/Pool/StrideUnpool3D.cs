namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>3次元ストライド逆プール</summary>
    public sealed class StrideUnpool3D : Unpool3D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public StrideUnpool3D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void strideunpool_3d(const float* __restrict__ inmap, float* __restrict__ outmap,
                                            unsigned int iz,
                                            unsigned int inwidth, unsigned int outwidth,
                                            unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY}, iy = {Defines.IndexZ};

                if (ch >= {Channels} || ix >= inwidth || iy >= inheight) {{
                    return;
                }}

                unsigned int ox = ix * {Stride}, oy = iy * {Stride}, oz = iz * {Stride};

                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + inheight * iz));
                unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + outheight * oz));

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "strideunpool_3d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint outwidth = (args[2] as uint?).Value;
            uint outheight = (args[3] as uint?).Value;
            uint outdepth = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint inwidth = outwidth / Stride, inheight = outheight / Stride, indepth = outdepth / Stride;

            outmap.ZerosetAsync(stream, Channels * outwidth * outheight * outdepth * batches);

            for (uint th = 0; th < batches; th++) {
                for (uint iz = 0; iz < indepth; iz++) {
                    Kernel.Execute(
                        indexes: (Channels, inwidth, inheight),
                        dynamic_shared_memory_bytes: 0,
                        stream,
                        inmap.ElementPtr(th * Channels * inwidth * inheight * indepth),
                        outmap.ElementPtr(th * Channels * outwidth * outheight * outdepth),
                        iz,
                        inwidth, outwidth, inheight, outheight
                    );
                }
            }
        }
    }
}
