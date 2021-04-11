using System;

namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>3次元最大値逆プール</summary>
    public sealed class MaxUnpool3D : Unpool3D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public MaxUnpool3D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void maxunpool_3d(const float* __restrict__ ingrad, const float* __restrict__ inpool, const float* __restrict__ inmap, float* __restrict__ outmap,
                                         unsigned int iz,
                                         unsigned int inwidth, unsigned int outwidth,
                                         unsigned int inheight, unsigned int outheight) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY}, iy = {Defines.IndexZ};

                if (ch >= {Channels} || ix >= inwidth || iy >= inheight) {{
                    return;
                }}

                unsigned int ox = ix * {Stride}, oy = iy * {Stride}, oz = iz * {Stride};

                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * (iy + inheight * iz));

                float g = ingrad[inmap_idx], v = inpool[inmap_idx];

                for(int kz = 0; kz < {Stride}; kz++){{
                    for(int ky = 0; ky < {Stride}; ky++){{
                        unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * ((oy + ky) + outheight * (oz + kz)));

                        for(int kx = 0; kx < {Stride}; kx++){{
                            float x = inmap[outmap_idx];

                            outmap[outmap_idx] = x >= v ? g : 0;
                            outmap_idx += {Channels};
                        }}
                    }}
                }}
            }}";

            this.Kernel = new Kernel(code, "maxunpool_3d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> ingrad = args[0] as CudaArray<float>;
            CudaArray<float> inpool = args[1] as CudaArray<float>;
            CudaArray<float> inmap = args[2] as CudaArray<float>;
            CudaArray<float> outmap = args[3] as CudaArray<float>;
            uint outwidth = (args[4] as uint?).Value;
            uint outheight = (args[5] as uint?).Value;
            uint outdepth = (args[6] as uint?).Value;
            uint batches = (args[7] as uint?).Value;

            uint inwidth = outwidth / Stride, inheight = outheight / Stride, indepth = outdepth / Stride;

            if (outwidth % Stride != 0 || outheight % Stride != 0 || outdepth % Stride != 0) {
                outmap.ZerosetAsync(stream, Channels * outwidth * outheight * outdepth * batches);
            }

            for (uint th = 0; th < batches; th++) {
                for (uint iz = 0; iz < indepth; iz++) {
                    Kernel.Execute(
                        indexes: (Channels, inwidth, inheight),
                        dynamic_shared_memory_bytes: 0,
                        stream,
                        ingrad.ElementPtr(th * Channels * inwidth * inheight * indepth),
                        inpool.ElementPtr(th * Channels * inwidth * inheight * indepth),
                        inmap.ElementPtr(th * Channels * outwidth * outheight * outdepth),
                        outmap.ElementPtr(th * Channels * outwidth * outheight * outdepth),
                        iz,
                        inwidth, outwidth, inheight, outheight
                    );
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 8) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[4] is uint outwidth) || !Limits.CheckWidth(outwidth, Stride)) {
                throw new ArgumentException(nameof(outwidth));
            }

            if (!(args[5] is uint outheight) || !Limits.CheckHeight(outheight, Stride)) {
                throw new ArgumentException(nameof(outheight));
            }

            if (!(args[6] is uint outdepth) || !Limits.CheckDepth(outdepth, Stride)) {
                throw new ArgumentException(nameof(outdepth));
            }

            if (!(args[7] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint inwidth = outwidth / Stride, inheight = outheight / Stride, indepth = outdepth / Stride;

            if (!(args[0] is CudaArray<float> ingrad) || ingrad.Length < Channels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException(nameof(ingrad));
            }

            if (!(args[1] is CudaArray<float> inpool) || inpool.Length < Channels * inwidth * inheight * indepth * batches) {
                throw new ArgumentException(nameof(inpool));
            }

            if (!(args[2] is CudaArray<float> inmap) || inmap.Length < Channels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[3] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * outdepth * batches) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
