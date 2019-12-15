using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>チャネル方向に切り出し</summary>
    internal class ExtractChannel : Operator {
        /// <summary>挿入先チャネル</summary>
        public int ChannelIndex { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ExtractChannel(Shape inshape, int channel_index, Shape outshape) {
            if (inshape.Type != ShapeType.Map) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", inshape));
            }
            if (outshape.Type != ShapeType.Map) {
                throw new ArgumentException(ExceptionMessage.Shape("Type", outshape));
            }

            if (channel_index < 0 || channel_index >= inshape.Channels || outshape.Channels + channel_index > inshape.Channels) {
                throw new ArgumentOutOfRangeException(nameof(channel_index));
            }

            if (!((int[])inshape).Skip(1).SequenceEqual(((int[])outshape).Skip(1))) {
                throw new ArgumentException(ExceptionMessage.Shape("MapLength", outshape));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, inshape),
                (ArgumentType.Out, outshape),
            };

            this.ChannelIndex = channel_index;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.ArrayManipulation.PatternCopy((uint)inmap.Channels, (uint)ChannelIndex, (uint)outmap.Channels, 0,
                                                           (uint)outmap.Channels, (uint)(outmap.Length / outmap.Channels), inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
