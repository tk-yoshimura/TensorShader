using System;

namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>2次元最大値逆プール</summary>
    public sealed class MaxUnpool2D : Unpool2D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public MaxUnpool2D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void maxunpool2d(float *ingrad, float *inpool, float *inmap, float *outmap, 
                                        unsigned int inwidth, unsigned int outwidth, 
                                        unsigned int inheight) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY}, iy = {Defines.IndexZ};

                if (ch >= {Channels} || ix >= inwidth || iy >= inheight) {{
                    return;
                }}

                unsigned int ox = ix * {Stride}, oy = iy * {Stride};

                unsigned int inmap_idx = ch + {Channels} * (ix + inwidth * iy);
                
                float g = ingrad[inmap_idx], v = inpool[inmap_idx];
                
                for(int ky = 0; ky < {Stride}; ky++){{
                    unsigned int outmap_idx = ch + {Channels} * (ox + outwidth * (oy + ky));
                    
                    for(int kx = 0; kx < {Stride}; kx++){{
                        float x = inmap[outmap_idx];

                        outmap[outmap_idx] = x >= v ? g : 0; 
                        outmap_idx += {Channels};
                    }}
                }}
            }}";

            this.Kernel = new Kernel(code, "maxunpool2d");
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
            uint batches = (args[6] as uint?).Value;

            uint inwidth = outwidth / Stride, inheight = outheight / Stride;

            if(outwidth % Stride != 0 || outheight % Stride != 0) { 
                outmap.ZerosetAsync(stream, Channels * outwidth * outheight * batches);
            }

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute((Channels, inwidth, inheight),
                    dynamic_shared_memory_bytes: 0, stream,
                    ingrad.ElementPtr(th * Channels * inwidth * inheight), 
                    inpool.ElementPtr(th * Channels * inwidth * inheight), 
                    inmap.ElementPtr(th * Channels * outwidth * outheight), 
                    outmap.ElementPtr(th * Channels * outwidth * outheight),
                    inwidth, outwidth, inheight
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 7) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[4] is uint outwidth) || outwidth < Stride) {
                throw new ArgumentException($"{nameof(args)}[4]");
            }

            if (!(args[5] is uint outheight) || outheight < Stride) {
                throw new ArgumentException($"{nameof(args)}[5]");
            }

            if (!(args[6] is uint batches) || batches < 1) {
                throw new ArgumentException($"{nameof(args)}[6]");
            }

            uint inwidth = outwidth / Stride, inheight = outheight / Stride;

            if (!(args[0] is CudaArray<float> ingrad) || ingrad.Length < Channels * inwidth * inheight * batches) {
                throw new ArgumentException($"{nameof(args)}[0]");
            }

            if (!(args[1] is CudaArray<float> inpool) || inpool.Length < Channels * inwidth * inheight * batches) {
                throw new ArgumentException($"{nameof(args)}[1]");
            }

            if (!(args[2] is CudaArray<float> inmap) || inmap.Length < Channels * outwidth * outheight * batches) {
                throw new ArgumentException($"{nameof(args)}[2]");
            }

            if (!(args[3] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * outheight * batches) {
                throw new ArgumentException($"{nameof(args)}[3]");
            }
        }
    }
}
