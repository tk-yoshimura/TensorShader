﻿using System;
using System.Linq;

namespace TensorShaderCudaBackend.Shaders.Padding {

    /// <summary>1次元パディング基底クラス</summary>
    public abstract class Padding1D : Shader {

        /// <summary>チャネル数</summary>
        public uint Channels { private set; get; }

        /// <summary>パディング左幅</summary>
        public uint PadLeft { private set; get; }

        /// <summary>パディング右幅</summary>
        public uint PadRight { private set; get; }

        /// <summary>識別子</summary>
        public override sealed string Signature =>
            $"{GetType().Name.Split(',').Last()} {nameof(Channels)} = {Channels} " +
            $"{nameof(PadLeft)} = {PadLeft} {nameof(PadRight)} = {PadRight}";

        /// <summary>コンストラクタ</summary>
        public Padding1D(uint channels, uint pad_left, uint pad_right) {
            if (!Limits.CheckChannels(channels)) {
                throw new ArgumentException(null, nameof(channels));
            }

            this.Channels = channels;
            this.PadLeft = pad_left;
            this.PadRight = pad_right;
        }

        /// <summary>実行</summary>
        public override sealed void Execute(Stream stream, params object[] args) {
            CheckArgument(args);

            CudaArray<float> inmap = args[0] as CudaArray<float>;
            CudaArray<float> outmap = args[1] as CudaArray<float>;
            uint inwidth = (args[2] as uint?).Value;
            uint batches = (args[3] as uint?).Value;

            uint outwidth = inwidth + PadLeft + PadRight;

            for (uint th = 0; th < batches; th++) {
                Kernel.Execute(
                    indexes: (Channels, outwidth),
                    dynamic_shared_memory_bytes: 0,
                    stream,
                    inmap.ElementPtr(th * Channels * inwidth),
                    outmap.ElementPtr(th * Channels * outwidth),
                    inwidth, outwidth
                );
            }
        }

        /// <summary>引数チェック</summary>
        protected override sealed void CheckArgument(params object[] args) {
            if (args is null || args.Length != 4) {
                throw new ArgumentException(null, nameof(args));
            }

            if (args[2] is not uint inwidth || !Limits.CheckWidth(inwidth)) {
                throw new ArgumentException(nameof(inwidth));
            }

            if (args[3] is not uint batches || !Limits.CheckBatches(batches)) {
                throw new ArgumentException(nameof(batches));
            }

            uint outwidth = inwidth + PadLeft + PadRight;

            if (args[0] is not CudaArray<float> inmap || inmap.Length < Channels * inwidth * batches) {
                throw new ArgumentException(nameof(inmap));
            }

            if (args[1] is not CudaArray<float> outmap || outmap.Length < Channels * outwidth * batches) {
                throw new ArgumentException(nameof(outmap));
            }
        }
    }
}
