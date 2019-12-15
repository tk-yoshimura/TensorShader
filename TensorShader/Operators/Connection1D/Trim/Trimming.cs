using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection1D {
    /// <summary>トリミング</summary>
    internal class Trimming : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>トリミング左幅</summary>
        public int TrimLeft { private set; get; }

        /// <summary>トリミング右幅</summary>
        public int TrimRight { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Trimming(int inwidth, int channels, int trim, int batch = 1) :
            this(inwidth, channels, trim, trim, batch) { }

        /// <summary>コンストラクタ</summary>
        public Trimming(int inwidth, int channels, int trim_left, int trim_right, int batch = 1) {
            int outwidth = inwidth - trim_left - trim_right;

            if (trim_left < 0 || trim_right < 0) {
                throw new ArgumentException($"{nameof(trim_left)}, {nameof(trim_right)}");
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(channels, inwidth, batch)),
                (ArgumentType.Out, Shape.Map1D(channels, outwidth, batch)),
            };

            this.Channels = channels;

            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;

            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Trimming.Trimming1D((uint)Channels, (uint)outmap.Width, (uint)Batch,
                                                        (uint)TrimLeft, (uint)TrimRight, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
