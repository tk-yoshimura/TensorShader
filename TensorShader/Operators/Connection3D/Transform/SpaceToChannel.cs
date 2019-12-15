using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>空間次元をチャネル次元に展開</summary>
    internal class SpaceToChannel : Operator {
        /// <summary>入力チャネル数</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public int OutChannels { private set; get; }

        /// <summary>倍率</summary>
        public int Scale { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public SpaceToChannel(int inwidth, int inheight, int indepth, int inchannels, int scale, int batch = 1) {
            if (scale < 2 || inwidth % scale != 0 || inheight % scale != 0 || indepth % scale != 0) {
                throw new ArgumentException($"{nameof(scale)}, {nameof(inwidth)}, {nameof(inheight)}, {nameof(indepth)}");
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(inchannels, inwidth, inheight, indepth, batch)),
                (ArgumentType.Out, Shape.Map3D(inchannels * scale * scale * scale, inwidth / scale, inheight / scale, indepth / scale, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = inchannels * scale * scale * scale;

            this.Scale = scale;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Transform.SpaceToChannel3D((uint)InChannels, (uint)outmap.Width, (uint)outmap.Height, (uint)outmap.Depth, (uint)Batch, (uint)Scale, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
