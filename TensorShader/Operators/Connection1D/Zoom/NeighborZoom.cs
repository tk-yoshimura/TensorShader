namespace TensorShader.Operators.Connection1D {
    /// <summary>最近傍補間</summary>
    /// <remarks>倍率2固定</remarks>
    internal class NeighborZoom : Zoom {
        /// <summary>コンストラクタ</summary>
        public NeighborZoom(int inwidth, int channels, int batch = 1)
            : base(inwidth, channels, scale: 2, batch) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Zoom.NeighborZoom1D((uint)Channels, (uint)inmap.Width,
                                                        (uint)Batch, inmap.Buffer, outmap.Buffer);
        }
    }
}
