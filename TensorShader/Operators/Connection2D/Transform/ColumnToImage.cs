using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection2D {
    /// <summary>ColumnToImage変換</summary>
    internal class ColumnToImage : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelHeight { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ColumnToImage(int inwidth, int inheight, int channels, int kwidth, int kheight, int batch = 1) {
            if (kwidth < 1 || kwidth % 2 != 1) {
                throw new ArgumentException(null, nameof(kwidth));
            }
            if (kheight < 1 || kheight % 2 != 1) {
                throw new ArgumentException(null, nameof(kheight));
            }

            int outwidth = inwidth + kwidth - 1;
            int outheight = inheight + kheight - 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, new Shape(ShapeType.Column, kwidth * kheight, channels, inwidth, inheight, batch)),
                (ArgumentType.Out, Shape.Map2D(channels, outwidth, outheight, batch)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;

            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Transform.ColumnToImage2D(
                (uint)Channels, (uint)inmap.Width, (uint)inmap.Height,
                (uint)Batch, (uint)KernelWidth, (uint)KernelHeight, inmap.Buffer, outmap.Buffer
            );
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
