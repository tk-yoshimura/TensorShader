namespace TensorShader.Operators.Connection2D {
    /// <summary>線形補間</summary>
    /// <remarks>倍率2固定</remarks>
    internal class LinearZoom : Zoom {
        /// <summary>コンストラクタ</summary>
        public LinearZoom(int inwidth, int inheight, int channels, int batch = 1)
            : base("LinearZoom", inwidth, inheight, channels, scale: 2, batch) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Zoom.LinearZoom2D((uint)Channels, (uint)inmap.Width, (uint)inmap.Height,
                                                      (uint)Batch, inmap.Buffer, outmap.Buffer);
        }
    }
}
