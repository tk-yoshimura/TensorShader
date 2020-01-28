using System;
using System.Linq;

using static TensorShaderCudaBackend.ArrayManipulation;

namespace TensorShaderCudaBackend.Shaders.Quaternion.Convolution {

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
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 4, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth)) {
                throw new ArgumentException(nameof(kwidth));
            }

            this.InChannels = inchannels / 4;
            this.OutChannels = outchannels / 4;
            this.KernelWidth = kwidth;
            this.Transpose = transpose;

            this.BlockSize = Kernel.MinimizeGridsBlockSize((InChannels, OutChannels));

            string code = $@"

            {Defines.CtorFloat4}
            {Defines.FloatFloatFma}
            {Defines.FloatFloatFms}
            {Defines.FloatFloatHiLoAdd}
            {Defines.Quaternion.KernelProd}
            {Defines.Quaternion.AtomicAdd}

            __global__ void quaternion_kernelproduct_1d(float4 *inmap, float4 *outmap, float4 *filter,
                                                        unsigned int outwidth) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY};
                unsigned int ox_offset = {Defines.BlockIndexZ} * {BatchPixels};
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                __shared__ float4 us[{BlockSize.x}], vs[{BlockSize.y}];

                for(unsigned int kx = 0; kx < {KernelWidth}; kx++){{
                    unsigned int filter_index = (inch + {InChannels} * (outch + {OutChannels} * kx)) * 2;

                    float4 uv_hi = ctor_float4(0.0, 0.0, 0.0, 0.0), uv_lo = ctor_float4(0.0, 0.0, 0.0, 0.0);

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

                            float4 u = us[tidx];
                            float4 v = vs[tidy];

                            quaternion_kernelprod(uv_hi, uv_lo, {(Transpose ? "v, u" : "u, v")});

                        { (InChannels % BlockSize.x != 0 ? "}" : "") }
                        { (OutChannels % BlockSize.y != 0 ? "}" : "") }

                        __syncthreads();
                    }}

                    { (InChannels % BlockSize.x != 0 ? $"if(inch < {InChannels}){{" : "") }
                    { (OutChannels % BlockSize.y != 0 ? $"if(outch < {OutChannels}){{" : "") }

                        floatfloat_atomicadd(filter + filter_index, uv_hi, uv_lo);

                    { (InChannels % BlockSize.x != 0 ? "}" : "") }
                    { (OutChannels % BlockSize.y != 0 ? "}" : "") }
                }}
            }}";

            this.Kernel = new Kernel(code, "quaternion_kernelproduct_1d");
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

            CudaArray<float> dfloat_filter =
                CudaArrayReserver<float>.Request(stream, inmap.DeviceID, index: 0, InChannels * OutChannels * KernelWidth * 8);
            dfloat_filter.ZerosetAsync(stream, InChannels * OutChannels * KernelWidth * 8);

            uint xsets = (outwidth + BatchPixels - 1) / BatchPixels;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (InChannels, OutChannels, xsets),
                    block: (BlockSize.x, BlockSize.y, 1),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * InChannels * inwidth * 4),
                    outmap.ElementPtr(th * OutChannels * outwidth * 4),
                    dfloat_filter,
                    outwidth
                );
            }

            HorizontalAdd(InChannels * OutChannels * KernelWidth * 4, dfloat_filter, filter, stream);
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 5) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * batches * 4) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * batches * 4) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
