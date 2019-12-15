using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection1D {
    /// <summary>拡大</summary>
    internal abstract class Zoom : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>倍率</summary>
        public int Scale { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        protected Zoom(int inwidth, int channels, int scale, int batch = 1) {
            if (scale < 2) {
                throw new ArgumentException(nameof(scale));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(channels, inwidth, batch)),
                (ArgumentType.Out, Shape.Map1D(channels, inwidth * scale, batch)),
            };

            this.Channels = channels;

            this.Scale = scale;

            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
