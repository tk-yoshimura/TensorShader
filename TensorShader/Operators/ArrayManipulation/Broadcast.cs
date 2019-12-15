using System;
using System.Collections.Generic;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>ブロードキャスト</summary>
    internal class Broadcast : Operator {
        /// <summary>コンストラクタ</summary>
        public Broadcast(Shape inshape, Shape outshape) {
            if (BroadcastShape(inshape, outshape) != outshape) {
                throw new ArgumentException(ExceptionMessage.Broadcast(inshape, outshape));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, inshape),
                (ArgumentType.Out, outshape),
            };
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            uint src_length = 1, dst_length = 1, slides = 1;

            int i;
            for (i = 0; i < inmap.Ndim; i++) {
                if (inmap.Shape[i] == outmap.Shape[i]) {
                    src_length *= (uint)inmap.Shape[i];
                    dst_length *= (uint)outmap.Shape[i];
                }
                else {
                    break;
                }
            }
            for (; i < outmap.Ndim; i++) {
                if (i >= inmap.Ndim || inmap.Shape[i] == 1) {
                    dst_length *= (uint)outmap.Shape[i];
                }
                else {
                    break;
                }
            }
            for (; i < outmap.Ndim; i++) {
                slides *= (uint)outmap.Shape[i];
            }

            TensorShaderCudaBackend.ArrayManipulation.Broadcast(src_length, inmap.Buffer, dst_length, outmap.Buffer, slides);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }

        /// <summary>ブロードキャスト出力形状</summary>
        internal static Shape BroadcastShape(Shape inshape, Shape targetshape) {
            int[] outshape = (int[])inshape;
            int i;
            for (i = 0; i < outshape.Length; i++) {
                if (outshape[i] == targetshape[i]) {
                    continue;
                }
                else {
                    break;
                }
            }
            for (; i < outshape.Length; i++) {
                if (outshape[i] == 1) {
                    outshape[i] = targetshape[i];
                }
                else {
                    break;
                }
            }
            if (i >= outshape.Length) {
                outshape = targetshape;
            }

            return new Shape(targetshape.Type, outshape);
        }
    }
}
