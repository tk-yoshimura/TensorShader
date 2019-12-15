using System.Collections.Generic;

namespace TensorShader.Operators.Indexer {
    /// <summary>次元ごとに最大インデクスを抽出</summary>
    internal class ArgMax : Operator {
        /// <summary>チャネル</summary>
        public int Channels { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ArgMax(int channels, int batch = 1) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map0D(channels, batch)),
                (ArgumentType.Out, Shape.Vector(batch)),
            };

            this.Channels = channels;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Indexer.ArgMax((uint)outmap.Length, (uint)Channels, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
