namespace TensorShader.Operators.Connection3D {
    /// <summary>ゼロパディング</summary>
    internal class ZeroPadding : Padding {
        /// <summary>コンストラクタ</summary>
        public ZeroPadding(int inwidth, int inheight, int indepth, int channels, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_rear, int batch = 1)
            : base("ZeroPadding", inwidth, inheight, indepth, channels, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear, batch) { }

        /// <summary>コンストラクタ</summary>
        public ZeroPadding(int inwidth, int inheight, int indepth, int channels, int pad, int batch = 1)
            : base("ZeroPadding", inwidth, inheight, indepth, channels, pad, batch) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderCudaBackend.Padding.ZeroPadding3D((uint)Channels, (uint)inmap.Width, (uint)inmap.Height, (uint)inmap.Depth, (uint)Batch,
                                                          (uint)PadLeft, (uint)PadRight, (uint)PadTop, (uint)PadBottom, (uint)PadFront, (uint)PadRear, inmap.Buffer, outmap.Buffer);
        }
    }
}
