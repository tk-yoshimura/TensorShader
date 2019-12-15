using System;
using System.Linq;
using static TensorShader.VariableNode;

namespace TensorShader.Links.BinaryArithmetric {
    /// <summary>2項演算</summary>
    public abstract class BinaryArithmetric : Link {
        /// <summary>入力項1</summary>
        protected Field X1 => InFields[0];

        /// <summary>入力項2</summary>
        protected Field X2 => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public BinaryArithmetric(Field infield1, Field infield2, Field outfield)
            : base(new Field[] { infield1, infield2 }, outfield) { }

        /// <summary>形状を調整する</summary>
        public static VariableNode AdjectShape(VariableNode node, Shape shape) {
            if (shape != node.Shape) {
                if (shape.Ndim == 0 && node.Shape.Ndim > 0) {
                    node = Sum(node, keepdims: false);
                }
                else if (shape.Ndim == 1 && node.Shape.Ndim > 1 && shape[0] == node.Shape[0]) {
                    int[] axes = (new int[node.Shape.Ndim - 1]).Select((_, idx) => idx + 1).ToArray();

                    node = Sum(node, axes, keepdims: false);
                }
                else {
                    throw new ArgumentException(ExceptionMessage.Broadcast(node.Shape, shape));
                }
            }

            return node;
        }

    }
}
