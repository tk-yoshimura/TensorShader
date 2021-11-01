using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection1D {
    /// <summary>ColumnToImage変換</summary>
    internal class ColumnToImage : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ColumnToImage(int inwidth, int channels, int kwidth, int batch = 1) {
            if (kwidth < 1 || kwidth % 2 != 1) {
                throw new ArgumentException(null, nameof(kwidth));
            }

            int outwidth = inwidth + kwidth - 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, new Shape(ShapeType.Column, kwidth, channels, inwidth, batch)),
                (ArgumentType.Out, Shape.Map1D(channels, outwidth, batch)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;

            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Transform.ColumnToImage1D(
                (uint)Channels, (uint)inmap.Width,
                (uint)Batch, (uint)KernelWidth, inmap.Buffer, outmap.Buffer
            );
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
