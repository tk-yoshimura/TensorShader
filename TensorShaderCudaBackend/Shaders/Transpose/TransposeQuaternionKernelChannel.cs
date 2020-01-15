using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Transpose {

    /// <summary>フィルタ行列入出力チャネル軸入れ替え</summary>
    public class TransposeQuaternionKernelChannel : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>実行あたりのポイント数(2^15=32768‬)</summary>
        public static uint PointsPerExecute => 0x8000;

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels}";

        /// <summary>コンストラクタ</summary>
        public TransposeQuaternionKernelChannel(uint inchannels, uint outchannels) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 4, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels / 4;
            this.OutChannels = outchannels / 4;

            string code = $@"

            __global__ void transpose_quaternion_kernel_channels(float4 *inmap, float4 *outmap,
                                                                 unsigned int pts) {{

                unsigned int inch = {Defines.IndexX}, outch = {Defines.IndexY}, i = {Defines.IndexZ};

                if (inch >= {InChannels} || outch >= {OutChannels} || i >= pts) {{
                    return;
                }}

                unsigned int inmap_idx = inch + {InChannels} * (outch + {OutChannels} * i);
                unsigned int outmap_idx = outch + {OutChannels} * (inch + {InChannels} * i);

                outmap[outmap_idx] = inmap[inmap_idx];
            }}";

            this.Kernel = new Kernel(code, "transpose_quaternion_kernel_channels");
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint pts = (args[2] as uint?).Value;

            for (uint p = 0; p < pts; p += PointsPerExecute) {
                uint pl = Math.Min(PointsPerExecute, pts - p);

                Kernel.Execute(
                    indexes: (InChannels, OutChannels, pl),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(p * InChannels * OutChannels * 4),
                    outmap.ElementPtr(p * InChannels * OutChannels * 4),
                    pl
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args == null || args.Length != 3) {
                throw new ArgumentException(nameof(args));
            }

            if (!(args[2] is uint pts) || pts < 1) {
                throw new ArgumentException(nameof(pts));
            }

            uint length = InChannels * OutChannels * pts * 4;

            if (!(args[0] is CudaArray<float> inmap) || inmap.Length < length) {
                throw new ArgumentException(nameof(inmap));
            }

            if (!(args[1] is CudaArray<float> outmap) || outmap.Length < length) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
