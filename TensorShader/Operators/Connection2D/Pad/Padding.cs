using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection2D {
    /// <summary>パディング</summary>
    internal abstract class Padding : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>パディング左幅</summary>
        public int PadLeft { private set; get; }

        /// <summary>パディング右幅</summary>
        public int PadRight { private set; get; }

        /// <summary>パディング上幅</summary>
        public int PadTop { private set; get; }

        /// <summary>パディング下幅</summary>
        public int PadBottom { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        protected Padding(string funcname, int inwidth, int inheight, int channels, int pad, int batch = 1)
            : this(funcname, inwidth, inheight, channels, pad, pad, pad, pad, batch) { }

        /// <summary>コンストラクタ</summary>
        protected Padding(string funcname, int inwidth, int inheight, int channels, int pad_left, int pad_right, int pad_top, int pad_bottom, int batch = 1) {
            if (pad_left < 0 || pad_right < 0 || pad_top < 0 || pad_bottom < 0) {
                throw new ArgumentException($"{nameof(pad_left)}, {nameof(pad_right)}, {nameof(pad_top)}, {nameof(pad_bottom)}");
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map2D(channels, inwidth, inheight, batch)),
                (ArgumentType.Out, Shape.Map2D(channels, inwidth + pad_left + pad_right, inheight + pad_top + pad_bottom, batch)),
            };

            this.Channels = channels;

            this.PadLeft = pad_left;
            this.PadRight = pad_right;
            this.PadTop = pad_top;
            this.PadBottom = pad_bottom;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
