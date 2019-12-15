using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.Indexer {
    /// <summary>インデクスをOneHot特徴量に変換</summary>
    internal class OneHotVector : Operator {
        /// <summary>チャネル</summary>
        public int Channels { private set; get; }

        /// <summary>コンストラクタ</summary>
        public OneHotVector(int channels, Shape inshape) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, inshape),
                (ArgumentType.Out, new Shape(ShapeType.Map, new int[] { channels }.Concat((int[])inshape).ToArray() ) ),
            };

            this.Channels = channels;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Indexer.OneHotVector((uint)inmap.Length, (uint)Channels, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
