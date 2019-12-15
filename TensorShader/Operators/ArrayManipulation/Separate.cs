using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>分離</summary>
    internal class Separate : Operator {
        private readonly uint in_stride;
        private readonly uint[] out_strides;

        /// <summary>コンストラクタ</summary>
        public Separate(Shape inshape, Shape[] outshapes, int axis) {
            if (!CheckShape(inshape, outshapes, axis)) {
                throw new ArgumentException(ExceptionMessage.Separate(axis, inshape, outshapes));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>();
            this.arguments.Add((ArgumentType.In, inshape));
            this.arguments.AddRange(outshapes.Select((shape) => (ArgumentType.Out, shape)));

            this.out_strides = new uint[outshapes.Length];
            this.in_stride = 0;

            for (int i = 0; i < outshapes.Length; i++) {
                out_strides[i] = 1;
                for (int j = 0; j <= axis; j++) {
                    out_strides[i] *= (uint)outshapes[i][j];
                }

                in_stride += out_strides[i];
            }
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor intensor = tensors.First();

            uint index = 0, slides = (uint)intensor.Length / in_stride;

            for (int i = 1; i < tensors.Length; i++) {
                Tensor outtensor = tensors[i];

                TensorShaderCudaBackend.ArrayManipulation.PatternCopy(in_stride, index, out_strides[i - 1], 0, out_strides[i - 1],
                                                               slides, intensor.Buffer, outtensor.Buffer);

                index += out_strides[i - 1];
            }
        }

        public static bool CheckShape(Shape inshape, Shape[] outshapes, int axis) {
            if (axis < 0 || axis >= inshape.Ndim) {
                return false;
            }

            int ndim = inshape.Ndim;
            int length = 0;

            foreach (Shape outshape in outshapes) {
                if (outshape.Ndim != ndim) {
                    return false;
                }

                length += outshape[axis];

                for (int i = 0; i < ndim; i++) {
                    if (i == axis) {
                        continue;
                    }

                    if (outshape[i] != inshape[i]) {
                        return false;
                    }
                }
            }

            if (length != inshape[axis]) {
                return false;
            }

            return true;
        }
    }
}
