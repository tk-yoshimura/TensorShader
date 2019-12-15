using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>定数に初期化</summary>
    internal class Fill : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>初期化する値</summary>
        public float FillValue { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Fill(Shape shape, float fill_val) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.Reference, shape),
            };

            this.Shape = shape;
            this.FillValue = fill_val;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor refmap = tensors[0];

            refmap.Clear(FillValue);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor refmap) {
            Execute(new Tensor[] { refmap });
        }
    }
}
