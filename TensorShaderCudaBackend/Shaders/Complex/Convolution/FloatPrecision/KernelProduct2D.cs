﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Complex.Convolution.FloatPrecision {

    /// <summary>カーネル積</summary>
    public sealed class KernelProduct2D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>実行あたりの積数(2^29=536870912)</summary>
        public static ulong MulPerExecute => 0x20000000;

        /// <summary>実行あたりのポイント数(2^14=16384‬)</summary>
        public static uint PointsPerExecute => 0x4000;

        /// <summary>ブロックサイズ</summary>
        public (uint x, uint y) BlockSize { private set; get; }

        /// <summary>1スレッドで処理する対象ピクセル数</summary>
        private static uint BatchPixels => 16;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight} {nameof(Transpose)} = {Transpose}";

        /// <summary>コンストラクタ</summary>
        public KernelProduct2D(uint inchannels, uint outchannels, uint kwidth, uint kheight, bool transpose) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 2, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}");
            }

            this.InChannels = inchannels / 2;
            this.OutChannels = outchannels / 2;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.Transpose = transpose;

            this.BlockSize = Kernel.MinimizeGridsBlockSize((InChannels, OutChannels));

            string code = $@"

            {Defines.CtorFloat2}
            {Defines.Float.Fma}
            {Defines.Float.Fms}
            {Defines.Float.Complex.KernelProd}
            {Defines.Float.Complex.AtomicAdd}

            __global__ void complex_kernelproduct_2d(const float2* __restrict__ inmap, const float2* __restrict__ outmap, float2* __restrict__ filter,
                                                     unsigned int oy_offset,
                                                     unsigned int xsets,
                                                     unsigned int inwidth, unsigned int outwidth) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY};
                unsigned int ox_offset = ({Defines.BlockIndexZ} % xsets) * {BatchPixels};
                unsigned int oy = oy_offset + {Defines.BlockIndexZ} / xsets;
                unsigned int tidx = {Defines.ThreadIdX}, tidy = {Defines.ThreadIdY};

                __shared__ float2 us[{BlockSize.x}], vs[{BlockSize.y}];

                for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{
                    for(unsigned int kx = 0; kx < {KernelWidth}; kx++){{
                        unsigned int filter_index = inch + {InChannels} * (outch + {OutChannels} * (kx + {KernelWidth} * ky));

                        float2 uv = ctor_float2(0.0, 0.0);

                        for(unsigned int ox = ox_offset, ix = ox + kx; ox < ox_offset + {BatchPixels} && ox < outwidth; ox++, ix++){{
                            { (OutChannels % BlockSize.y != 0 ? $"if(tidx == 0 && outch < {OutChannels}){{" : "if(tidx == 0){") }
                                vs[tidy] = outmap[outch + {OutChannels} * (ox + outwidth * oy)];
                            }}
                            { (InChannels % BlockSize.x != 0 ? $"if(tidy == 0 && inch < {InChannels}){{" : "if(tidy == 0){") }
                                us[tidx] = inmap[inch + {InChannels} * (ix + inwidth * iy)];
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
                }}
            }}";

            this.Kernel = new Kernel(code, "complex_kernelproduct_2d");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint inwidth = (args[3] as uint?).Value;
            uint inheight = (args[4] as uint?).Value;
            uint batches = (args[5] as uint?).Value;

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;

            filter.ZerosetAsync(stream, InChannels * OutChannels * KernelWidth * KernelHeight * 2);

            ulong mul_per_line = (ulong)InChannels * OutChannels * KernelWidth * KernelHeight * outwidth * 4;

            uint lines_per_execute_mul = (uint)(MulPerExecute / mul_per_line + 1);
            uint lines_per_execute_pixels = (PointsPerExecute + outwidth - 1) / outwidth;

            uint lines_per_execute = Math.Min(lines_per_execute_mul, lines_per_execute_pixels);

            uint xsets = (outwidth + BatchPixels - 1) / BatchPixels;

            for (uint th = 0; th < batches; th++) {
                for (uint oy_offset = 0; oy_offset < outheight; oy_offset += lines_per_execute) {
                    uint lines = Math.Min(lines_per_execute, outheight - oy_offset);

                    Kernel.Execute(
                        indexes: (InChannels, OutChannels, xsets * lines),
                        block: (BlockSize.x, BlockSize.y, 1),
                        dynamic_shared_memory_bytes: 0,
                        stream,
                        inmap.ElementPtr(th * InChannels * inwidth * inheight * 2),
                        outmap.ElementPtr(th * OutChannels * outwidth * outheight * 2),
                        filter,
                        oy_offset,
                        xsets,
                        inwidth, outwidth
                    );

                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 6) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[3] is not uint inwidth || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[4] is not uint inheight || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (args[5] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < InChannels * inwidth * inheight * batches * 2) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < OutChannels * outwidth * outheight * batches * 2) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight * 2) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
