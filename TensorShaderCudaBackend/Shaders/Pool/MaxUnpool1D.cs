using System;

namespace TensorShaderCudaBackend.Shaders.Pool {

    /// <summary>1次元最大値逆プール</summary>
    public sealed class MaxUnpool1D : Unpool1D {

        /// <summary>コンストラクタ</summary>
        /// <param name="channels">チャネル数</param>
        /// <param name="stride">ストライド</param>
        public MaxUnpool1D(uint channels, uint stride)
            : base(channels, stride) {

            string code = $@"

            __global__ void maxunpool_1d(float *ingrad, float *inpool, float *inmap, float *outmap, 
                                         unsigned int inwidth) {{

                unsigned int ch = {Defines.IndexX}, ix = {Defines.IndexY};

                if (ch >= {Channels} || ix >= inwidth) {{
                    return;
                }}

                unsigned int ox = ix * {Stride};

                unsigned int inmap_idx = ch + {Channels} * ix;
                unsigned int outmap_idx = ch + {Channels} * ox;
                
                float g = ingrad[inmap_idx], v = inpool[inmap_idx];

                for(int kx = 0; kx < {Stride}; kx++){{
                    float x = inmap[outmap_idx];

                    outmap[outmap_idx] = x >= v ? g : 0; 
                    outmap_idx += {Channels};
                }}
            }}";

            this.Kernel = new Kernel(code, "maxunpool_1d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> ingrad = args[0] as CudaArray<float>;
            CudaArray<float> inpool = args[1] as CudaArray<float>;
            CudaArray<float> inmap = args[2] as CudaArray<float>;
            CudaArray<float> outmap = args[3] as CudaArray<float>;
            uint outwidth = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint inwidth = outwidth / Stride;

            if(outwidth % Stride != 0) { 
                outmap.ZerosetAsync(stream, Channels * outwidth * batches);
            }

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes:(Channels, inwidth),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    ingrad.ElementPtr(th * Channels * inwidth), 
                    inpool.ElementPtr(th * Channels * inwidth), 
                    inmap.ElementPtr(th * Channels * outwidth), 
                    outmap.ElementPtr(th * Channels * outwidth),
                    inwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 6) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[4] is uint outwidth) || !Limits.CheckWidth(outwidth, Stride)) {
                throw new ArgumentException(nameof(outwidth));
            }

            if (!(args[5] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint inwidth = outwidth / Stride;

            if (!(args[0] is CudaArray<float> ingrad) || ingrad.Length < Channels * inwidth * batches) {
                throw new ArgumentException(nameof(ingrad));
            }

            if (!(args[1] is CudaArray<float> inpool) || inpool.Length < Channels * inwidth * batches) {
                throw new ArgumentException(nameof(inpool));
            }

            if (!(args[2] is CudaArray<float> inmap) || inmap.Length < Channels * outwidth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[3] is CudaArray<float> outmap) || outmap.Length < Channels * outwidth * batches) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
