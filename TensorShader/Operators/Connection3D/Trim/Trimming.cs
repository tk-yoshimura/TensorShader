using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
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

        /// <summary>トリミング前幅</summary>
        public int TrimFront { private set; get; }

        /// <summary>トリミング後幅</summary>
        public int TrimRear { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Trimming(int inwidth, int inheight, int indepth, int channels, int trim, int batch = 1) :
            this(inwidth, inheight, indepth, channels, trim, trim, trim, trim, trim, trim, batch) { }

        /// <summary>コンストラクタ</summary>
        public Trimming(int inwidth, int inheight, int indepth, int channels, int trim_left, int trim_right, int trim_top, int trim_bottom, int trim_front, int trim_rear, int batch = 1) {
            int outwidth = inwidth - trim_left - trim_right;
            int outheight = inheight - trim_top - trim_bottom;
            int outdepth = indepth - trim_front - trim_rear;

            if (trim_left < 0 || trim_right < 0 || trim_top < 0 || trim_bottom < 0 || trim_front < 0 || trim_rear < 0) {
                throw new ArgumentException($"{nameof(trim_left)}, {nameof(trim_right)}, {nameof(trim_top)}, {nameof(trim_bottom)}");
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(channels, inwidth, inheight, indepth, batch)),
                (ArgumentType.Out, Shape.Map3D(channels, outwidth, outheight, outdepth, batch)),
            };

            this.Channels = channels;

            this.TrimLeft = trim_left;
            this.TrimRight = trim_right;
            this.TrimTop = trim_top;
            this.TrimBottom = trim_bottom;
            this.TrimFront = trim_front;
            this.TrimRear = trim_rear;

            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Trimming.Trimming3D((uint)Channels, (uint)outmap.Width, (uint)outmap.Height, (uint)outmap.Depth,
                                                        (uint)Batch,
                                                        (uint)TrimLeft, (uint)TrimRight, (uint)TrimTop, (uint)TrimBottom, (uint)TrimFront, (uint)TrimRear,
                                                        inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
