namespace TensorShader.Operators.Connection1D {
    /// <summary>ゼロパディング</summary>
    internal class ZeroPadding : Padding {
        /// <summary>コンストラクタ</summary>
        public ZeroPadding(int inwidth, int channels, int pad_left, int pad_right, int batch = 1)
            : base(inwidth, channels, pad_left, pad_right, batch) { }

        /// <summary>コンストラクタ</summary>
        public ZeroPadding(int inwidth, int channels, int pad, int batch = 1)
            : base(inwidth, channels, pad, batch) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];


            TensorShaderCudaBackend.Padding.ZeroPadding1D((uint)Channels, (uint)inmap.Width, (uint)Batch,
                                                          (uint)PadLeft, (uint)PadRight, inmap.Buffer, outmap.Buffer);
        }
    }
}
