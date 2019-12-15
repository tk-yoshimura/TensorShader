using System;
using System.Collections.Generic;

namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>ベクトル化2項演算</summary>
    internal abstract class BinaryRightVectorArithmetric : Operator {
        /// <summary>ベクトル形状</summary>
        public Shape VectorShape { private set; get; }

        /// <summary>マップ形状</summary>
        public Shape MapShape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public BinaryRightVectorArithmetric(Shape vectorshape, Shape mapshape) {
            if (vectorshape.Ndim >= mapshape.Ndim) {
                throw new ArgumentException(ExceptionMessage.Vectorize(vectorshape, mapshape));
            }

            for (int i = 0; i < vectorshape.Ndim; i++) {
                if (vectorshape[i] != mapshape[i]) {
                    throw new ArgumentException(ExceptionMessage.Vectorize(vectorshape, mapshape));
                }
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, mapshape),
                (ArgumentType.In, vectorshape),
                (ArgumentType.Out, mapshape),
            };

            this.VectorShape = vectorshape;
            this.MapShape = mapshape;
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outmap) {
            Execute(new Tensor[] { inmap1, inmap2, outmap });
        }
    }
}
