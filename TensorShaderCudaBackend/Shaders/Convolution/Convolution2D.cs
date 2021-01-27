using System;
using System.Linq;

using static TensorShaderCudaBackend.Transpose;

namespace TensorShaderCudaBackend.Shaders.Convolution {

    /// <summary>2次元畳み込み</summary>
    public sealed class Convolution2D : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        public uint KernelHeight { private set; get; }

        /// <summary>実行あたりの積数(2^30=1073741824‬)</summary>
        public static ulong MulPerExecute => 0x40000000;

        /// <summary>タイルサイズ</summary>
        public static uint TileSize => 8;

        /// <summary>Xスレッド数</summary>
        private uint ThreadsX { set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(KernelWidth)} = {KernelWidth} {nameof(KernelHeight)} = {KernelHeight}";

        /// <summary>コンストラクタ</summary>
        public Convolution2D(uint inchannels, uint outchannels, uint kwidth, uint kheight) {
            if (!Limits.CheckChannels(inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }
            if (!Limits.CheckKernelSize(kwidth, kheight)) {
                throw new ArgumentException($"{nameof(kwidth)}, {nameof(kheight)}");
            }

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;

            this.ThreadsX = Kernel.DefaultBlockSize(OutChannels);

            string code = $@"

            {Defines.FloatFloatFma}
            {Defines.StoreFloatSharedMemory(elemsize: 1, InChannels, ThreadsX)}

            __global__ void convolution_2d(const float* __restrict__ inmap, float* __restrict__ outmap, const float* __restrict__ filter,
                                           unsigned int tile_offset, unsigned tile_xs,
                                           unsigned int inwidth, unsigned int outwidth, unsigned int outheight) {{

                unsigned int outch = {Defines.IndexX}, tid = {Defines.ThreadIdX};
                unsigned int tile_index = {Defines.BlockIndexY}, tile_id = tile_offset + {Defines.BlockIndexZ};
                unsigned int tile_x = tile_id % tile_xs, tile_y = tile_id / tile_xs;

                unsigned int ox = tile_x * {TileSize} + tile_index % {TileSize};
                unsigned int oy = tile_y * {TileSize} + tile_index / {TileSize};
 
                if(ox >= outwidth || oy >= outheight){{
                    return;
                }}

                __shared__ float us[{InChannels}];
                float uv_hi = 0.0, uv_lo = 0.0;

                unsigned int filter_idx = outch;

                for(unsigned int ky = 0, iy = oy; ky < {KernelHeight}; ky++, iy++){{

                    unsigned int inmap_idx = {InChannels} * (ox + inwidth * iy);

                    { (KernelWidth <= 7 ? "#pragma unroll" : "") }
                    for(unsigned int kx = 0, ix = ox; kx < {KernelWidth}; kx++, ix++){{

                        store_smem(inmap + inmap_idx, us, tid);
                        inmap_idx += {InChannels};

                        { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                            #pragma unroll 8
                            for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                                float u = us[inch];
                                float v = filter[filter_idx];

                                floatfloat_fma(uv_hi, uv_lo, u, v);

                                filter_idx += {OutChannels};
                            }}

                        { (OutChannels % ThreadsX != 0 ? "}" : "") }
                        __syncthreads();
                    }}
                }}

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }
                    unsigned int outmap_idx = outch + {OutChannels} * (ox + outwidth * oy);

                    outmap[outmap_idx] = uv_hi + uv_lo;
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "convolution_2d");
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

            uint tile_xs = (outwidth + TileSize - 1) / TileSize;
            uint tile_ys = (outheight + TileSize - 1) / TileSize;
            uint tile_counts = tile_xs * tile_ys;

            ulong mul_per_tile = (ulong)InChannels * OutChannels * KernelWidth * KernelHeight * TileSize * TileSize;

            uint tiles_per_execute = (uint)(MulPerExecute / mul_per_tile + 1);

            CudaArray<float> transpose_filter =
                WorkspaceReserver<float>.Request(stream, filter.DeviceID, index: 0, InChannels * OutChannels * KernelWidth * KernelHeight);

            TransposeKernelChannel(InChannels, OutChannels, KernelWidth * KernelHeight, filter, transpose_filter, stream);

            for (uint th = 0; th < batches; th++) {
                for (uint tile_offset = 0; tile_offset < tile_counts; tile_offset += tiles_per_execute) {
                    uint tiles = Math.Min(tiles_per_execute, tile_counts - tile_offset);

                    Kernel.Execute(
                        indexes: (OutChannels, TileSize * TileSize, tiles),
                        block: (ThreadsX, 1, 1),
                        dynamic_shared_memory_bytes: 0, stream,
                        inmap.ElementPtr(th * InChannels * inwidth * inheight),
                        outmap.ElementPtr(th * OutChannels * outwidth * outheight),
                        transpose_filter,
                        tile_offset, tile_xs,
                        inwidth, outwidth, outheight
                    );
                }
            }
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args == null || args.Length != 6) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[3] is uint inwidth) || !Limits.CheckWidth(inwidth, KernelWidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (!(args[4] is uint inheight) || !Limits.CheckHeight(inheight, KernelHeight)) {
                throw new ArgumentException(nameof(inheight));
            }

            if (!(args[5] is uint batches) || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + 1 - KernelWidth;
            uint outheight = inheight + 1 - KernelHeight;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < InChannels * inwidth * inheight * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < OutChannels * outwidth * outheight * batches) {
                throw new ArgumentException(nameof(outmap));
            }

            if (!(args[2] is CudaArray<float> filter) || filter.Length < InChannels * OutChannels * KernelWidth * KernelHeight) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
