using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection1D {
    /// <summary>パディング</summary>
    internal abstract class Padding : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>パディング左幅</summary>
        public int PadLeft { private set; get; }

        /// <summary>パディング右幅</summary>
        public int PadRight { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        protected Padding(string funcname, int inwidth, int channels, int pad, int batch = 1)
            : this(inwidth, channels, pad, pad, batch) { }

        /// <summary>コンストラクタ</summary>
        protected Padding(int inwidth, int channels, int pad_left, int pad_right, int batch = 1) {
            if (pad_left < 0 || pad_right < 0) {
                throw new ArgumentException($"{nameof(pad_left)}, {nameof(pad_right)}");
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(channels, inwidth, batch)),
                (ArgumentType.Out, Shape.Map1D(channels, inwidth + pad_left + pad_right, batch)),
            };

            this.Channels = channels;

            this.PadLeft = pad_left;
            this.PadRight = pad_right;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
