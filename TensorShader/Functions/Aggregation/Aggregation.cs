using System;

namespace TensorShader.Functions.Aggregation {
    /// <summary>統計</summary>
    internal abstract class Aggregation : Function {
        /// <summary>軸</summary>
        public int Axis { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Aggregation(int axis)
            : base(inputs: 1, outputs: 1, allow_resubstitution: false) {

            this.Axis = axis;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape shape = inshapes[0];

            int[] lengths = shape;
            lengths[Axis] = 1;

            return new Shape[] { new Shape(shape.Type, lengths) };
        }

        /// <summary>入力テンソル形状をチェックする</summary>
        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (Axis >= inshapes[0].Ndim) {
                throw new ArgumentOutOfRangeException(nameof(Axis));
            }
        }
    }
}
