using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>一切の操作を行わない</summary>
    internal class None : Operator {
        /// <summary>コンストラクタ</summary>
        public None() {
            this.arguments = new List<(ArgumentType type, Shape shape)>();
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);
        }

        /// <summary>操作を実行</summary>
        public void Execute() { }
    }
}
