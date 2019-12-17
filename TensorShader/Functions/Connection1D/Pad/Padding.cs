using System;

namespace TensorShader.Functions.Connection1D {
    /// <summary>パディング</summary>
    internal abstract class Padding : Function {
        /// <summary>パディング左幅</summary>
        public int PadLeft { private set; get; }

        /// <summary>パディング右幅</summary>
        public int PadRight { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Padding(int pad)
            : this(pad, pad) { }

        /// <summary>コンストラクタ</summary>
        public Padding(int pad_left, int pad_right)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.PadLeft = pad_left;
            this.PadRight = pad_right;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            Shape outshape = Shape.Map1D(inshape.Channels,
                                         inshape.Width + PadLeft + PadRight,
                                         inshape.Batch);

            return new Shape[] { outshape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Type != ShapeType.Map || inshapes[0].Ndim != 3) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshapes[0], ("Ndim", 3), ("Type", ShapeType.Map)));
            }
        }
    }
}
