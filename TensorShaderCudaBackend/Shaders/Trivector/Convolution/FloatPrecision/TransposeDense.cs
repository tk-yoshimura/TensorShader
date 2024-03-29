﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Trivector.Convolution.FloatPrecision {

    /// <summary>転置全結合</summary>
    public sealed class TransposeDense : Shader {

        /// <summary>入力チャネル数</summary>
        public uint InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public uint OutChannels { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>Xスレッド数</summary>
        private uint ThreadsX { set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(InChannels)} = {InChannels} {nameof(OutChannels)} = {OutChannels} " +
            $"{nameof(GradMode)} = {GradMode}";

        /// <summary>コンストラクタ</summary>
        public TransposeDense(uint inchannels, uint outchannels, bool gradmode) {
            if (!Limits.CheckChannels(inchannels, outchannels) || !Limits.CheckMultipleNum(multiple: 3, inchannels, outchannels)) {
                throw new ArgumentException($"{nameof(inchannels)}, {nameof(outchannels)}");
            }

            this.InChannels = inchannels / 3;
            this.OutChannels = outchannels / 3;
            this.GradMode = gradmode;

            this.ThreadsX = Kernel.DefaultBlockSize(OutChannels);

            string code = $@"

            {Defines.CtorFloat3}
            {Defines.Float.Fma}
            {Defines.Float.Trivector.Mul}
            {Defines.Float.Trivector.MulGrad}
            {Defines.StoreFloatSharedMemory(elemsize: 3, InChannels, ThreadsX)}

            __global__ void trivector_transpose_dense(const float3* __restrict__ inmap, float3* __restrict__ outmap, const float4* __restrict__ filter) {{

                unsigned int outch = {Defines.IndexX}, th = {Defines.BlockIndexY};
                unsigned int tid = {Defines.ThreadIdX};

                unsigned int inmap_offset = {InChannels} * th;
                inmap += inmap_offset;

                unsigned int outmap_offset = {OutChannels} * th;
                outmap += outmap_offset;

                __shared__ float3 vs[{InChannels}];
                float3 vq = ctor_float3(0.0, 0.0, 0.0);

                unsigned int filter_idx = outch;

                store_smem(inmap, vs, tid);

                { (OutChannels % ThreadsX != 0 ? $"if(outch < {OutChannels}){{" : "") }

                    #pragma unroll 4
                    for(unsigned int inch = 0; inch < {InChannels}; inch++){{
                        float3 v = vs[inch];
                        float4 q = filter[filter_idx];

                        {(GradMode ? "trivector_quaternion_mulgrad" : "trivector_quaternion_mul")}(vq, v, q);

                        filter_idx += {OutChannels};
                    }}

                    outmap[outch] = vq;
                { (OutChannels % ThreadsX != 0 ? "}" : "") }
            }}";

            this.Kernel = new Kernel(code, "trivector_transpose_dense");
        }

        /// <summary>実行</summary>
        public override void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            CudaArray<float> filter = args[2] as CudaArray<float>;

            uint batches = (args[3] as uint?).Value;

            Kernel.Execute(
                indexes: (OutChannels, batches),
                block: (ThreadsX, 1),
                dynamic_shared_memory_bytes: 0,
                stream,
                inmap,
                outmap,
                filter
            );
        }

        /// <summary>引数チェック</summary>
        protected override void CheckArgument(params object[] args) {
            if (args is null || args.Length != 4) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[3] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            if (args[0] is not CudaArray<float> inmap || inmap.Length < InChannels * batches * 3) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < OutChannels * batches * 3) {
                throw new ArgumentException(nameof(outmap));
            }

            if (args[2] is not CudaArray<float> filter || filter.Length < InChannels * OutChannels * 4) {
                throw new ArgumentException(nameof(filter));
            }
        }
    }
}
