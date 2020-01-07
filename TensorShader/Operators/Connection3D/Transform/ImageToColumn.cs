using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>ImageToColumn変換</summary>
    internal class ImageToColumn : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelHeight { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelDepth { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ImageToColumn(int inwidth, int inheight, int indepth, int channels, int kwidth, int kheight, int kdepth, int batch = 1) {
            if (kwidth < 1 || kwidth % 2 != 1) {
                throw new ArgumentException(nameof(kwidth));
            }
            if (kheight < 1 || kheight % 2 != 1) {
                throw new ArgumentException(nameof(kheight));
            }
            if (kdepth < 1 || kdepth % 2 != 1) {
                throw new ArgumentException(nameof(kdepth));
            }

            int outwidth = inwidth - kwidth + 1;
            int outheight = inheight - kheight + 1;
            int outdepth = indepth - kdepth + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(channels, inwidth, inheight, indepth, batch)),
                (ArgumentType.Out, new Shape(ShapeType.Column, kwidth * kheight * kdepth, channels, outwidth, outheight, outdepth, batch)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;

            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Transform.ImageToColumn3D(
                (uint)Channels, (uint)inmap.Width, (uint)inmap.Height, (uint)inmap.Depth,
                (uint)Batch, (uint)KernelWidth, (uint)KernelHeight, (uint)KernelDepth, inmap.Buffer, outmap.Buffer
            );
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
