using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection2D {
    /// <summary>トリミング</summary>
    internal class Trimming : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>トリミング左幅</summary>
        public int TrimLeft { private set; get; }

        /// <summary>トリミング右幅</summary>
        public int TrimRight { private set; get; }

        /// <summary>トリミング上幅</summary>
        public int TrimTop { private set; get; }

        /// <summary>トリミング下幅</summary>
        public int TrimBottom { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Trimming(int inwidth, int inheight, int channels, int trim, int batch = 1) :
            this(inwidth, inheight, channels, trim, trim, trim, trim, batch) { }

        /// <summary>コンストラクタ</summary>
        public Trimming(int inwidth, int inheight, int channels, int trim_left, int trim_right, int trim_top, int trim_bottom, int batch = 1) {
            int outwidth = inwidth - trim_left - trim_right;
            int outheight = inheight - trim_top - trim_bottom;

            if (trim_left < 0 || trim_right < 0 || trim_top < 0 || trim_bottom < 0) {
                throw new ArgumentException($"{nameof(trim_left)}, {nameof(trim_right)}, {nameof(trim_top)}, {nameof(trim_bottom)}");
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map2D(channels, inwidth, inheight, batch)),
                (ArgumentType.Out, Shape.Map2D(channels, outwidth, outheight, batch)),
            };

            this.Channels = channels;

            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;
            this.TrimTop = trim_top;
            this.TrimBottom = trim_bottom;

            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Trimming.Trimming2D((uint)Channels, (uint)outmap.Width, (uint)outmap.Height, (uint)Batch, 
                                                        (uint)TrimLeft, (uint)TrimRight, (uint)TrimTop, (uint)TrimBottom,
                                                        inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
