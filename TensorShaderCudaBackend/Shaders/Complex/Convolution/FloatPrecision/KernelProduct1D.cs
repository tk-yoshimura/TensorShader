using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Complex.Convolution.FloatPrecision {

    /// <summary>カーネル積</summary>
    public sealed class KernelProduct1D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>ブロックサイズ</summary>
        public (uint x, uint y) BlockSize { private set; get; }

        /// <summary>1スレッドで処理する対象ピクセル数</summary>
        private static uint BatchPixels => 16;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(Transpose)} = {Transpose}";

        /// <summary>コンストラクタ</summary>
        public KernelProduct1D(uint inchannels, uint outchannels, uint kwidth, bool transpose) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 2, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(null, nameof(kwidth));
            }

            this.InChannels = inchannels / 2;
            this.OutChannels = outchannels / 2;
            this.KernelWidth = kwidth;
            this.Transpose = transpose;

            this.BlockSize = Kernel.MinimizeGridsBlockSize((InChannels, OutChannels));

            string code = $@"

            {Defines.CtorFloat2}
            {Defines.Float.Fma}
            {Defines.Float.Fms}
            {Defines.Float.Complex.KernelProd}
            {Defines.Float.Complex.AtomicAdd}

            __global__ void complex_kernelproduct_1d(const float2* __restrict__ inmap, const float2* __restrict__ outmap, float2* __restrict__ filter,
                                                     unsigned int outwidth) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY};
                unsigned int ox_offset = {Defines.BlockIndexZ} * {BatchPixels};
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                __shared__ float2 us[{BlockSize.x}], vs[{BlockSize.y}];

                for(unsigned int kx = 0; kx < {KernelWidth}; kx++){{
                    unsigned int filter_index = inch + {InChannels} * (outch + {OutChannels} * kx);

                    float2 uv = ctor_float2(0.0, 0.0);

                    for(unsigned int ox = ox_offset, ix = ox + kx; ox < ox_offset + {BatchPixels} && ox < outwidth; ox++, ix++){{
                        { (OutChannels % BlockSize.y != 0 ? $"if(tidx == 0 && outch < {OutChannels}){{" : "if(tidx == 0){") }
                            vs[tidy] = outmap[outch + {OutChannels} * ox];
                        }}
                        { (InChannels % BlockSize.x != 0 ? $"if(tidy == 0 && inch < {InChannels}){{" : "if(tidy == 0){") }
                            us[tidx] = inmap[inch + {InChannels} * ix];
                        }}
                        __syncthreads();

                        { (InChannels % BlockSize.x != 0 ? $"if(inch < {InChannels}){{" : "") }
                        { (OutChannels % BlockSize.y != 0 ? $"if(outch < {OutChannels}){{" : "") }

                            float2 u = us[tidx];
                            float2 v = vs[tidy];

                            complex_kernelprod(uv, {(Transpose ? "v, u" : "u, v")});

                        { (InChannels % BlockSize.x != 0 ? "}" : "") }
                        { (OutChannels % BlockSize.y != 0 ? "}" : "") }

                        __syncthreads();
                    }}

                    { (InChannels % BlockSize.x != 0 ? $"if(inch < {InChannels}){{" : "") }
                    { (OutChannels % BlockSize.y != 0 ? $"if(outch < {OutChannels}){{" : "") }

                        float_atomicadd(filter + filter_index, uv);

                    { (InChannels % BlockSize.x != 0 ? "}" : "") }
                    { (OutChannels % BlockSize.y != 0 ? "}" : "") }
                }}
            }}";

            this.Kernel = new Kernel(code, "complex_kernelproduct_1d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint batches = (args[4] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;

            filter.ZerosetAsync(stream, InChannels * OutChannels * KernelWidth * 2);

            uint xsets = (outwidth + BatchPixels - 1) / BatchPixels;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (InChannels, OutChannels, xsets),
                    block: (BlockSize.x, BlockSize.y, 1),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * InChannels * inwidth * 2),
                    outmap.ElementPtr(th * OutChannels * outwidth * 2),
                    filter,
                    outwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 5) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[3] is not uint inwidth || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[4] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < InChannels * inwidth * batches * 2) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < OutChannels * outwidth * batches * 2) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < InChannels * OutChannels * KernelWidth * 2) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
