using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader {
    /// <summary>操作クラス</summary>
    public abstract class Operator {
        /// <summary>変数の入出力タイプおよび形状</summary>
        protected List<(ArgumentType type, Shape shape)> arguments;

        /// <summary>変数の入出力タイプおよび形状</summary>
        public IReadOnlyList<(ArgumentType type, Shape shape)> Arguments => arguments.AsReadOnly();

        /// <summary>変数の数</summary>
        public int ArgumentCount => arguments.Count;

        /// <summary>操作クラス名</summary>
        public string Name => GetType().Name.Split('.').Last();

        /// <summary>テンソル形状の有効性をチェック</summary>
        public void CheckArgumentShapes(params Tensor[] tensors) {
            if (arguments == null) {
                throw new NotImplementedException();
            }

            if (tensors.Length != ArgumentCount) {
                throw new ArgumentException(ExceptionMessage.ArgumentCount(nameof(tensors), tensors.Length, ArgumentCount));
            }

            for (int i = 0; i < ArgumentCount; i++) {
                if (tensors[i].Shape != arguments[i].shape) {
                    throw new ArgumentException(ExceptionMessage.ShapeWithIndex(i, tensors[i].Shape, arguments[i].shape));
                }
            }
        }

        /// <summary>操作を実行</summary>
        public abstract void Execute(params Tensor[] tensors);
    }
}
