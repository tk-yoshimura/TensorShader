using System;
using System.Linq;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>結合</summary>
    internal class Concat : Operator {
        private readonly uint out_stride;
        private readonly uint[] in_strides;

        /// <summary>コンストラクタ</summary>
        public Concat(Shape[] inshapes, Shape outshape, int axis) {
            if (!CheckShape(inshapes, outshape, axis)) {
                throw new ArgumentException(ExceptionMessage.Concat(axis, inshapes, outshape));
            }

            this.arguments = inshapes.Select((shape) => (ArgumentType.In, shape)).ToList();
            this.arguments.Add((ArgumentType.Out, outshape));

            this.in_strides = new uint[inshapes.Length];
            this.out_stride = 0;

            for (int i = 0; i < inshapes.Length; i++) {
                in_strides[i] = 1;
                for (int j = 0; j <= axis; j++) {
                    in_strides[i] *= (uint)inshapes[i][j];
                }

                out_stride += in_strides[i];
            }
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor outtensor = tensors.Last();

            uint index = 0, slides = (uint)outtensor.Length / out_stride;

            for (int i = 0; i < tensors.Length - 1; i++) {
                Tensor intensor = tensors[i];

                TensorShaderCudaBackend.ArrayManipulation.PatternCopy(in_strides[i], 0, out_stride, index, in_strides[i],
                                                               slides, intensor.Buffer, outtensor.Buffer);

                index += in_strides[i];
            }
        }

        public static bool CheckShape(Shape[] inshapes, Shape outshape, int axis) {
            if (axis < 0 || axis >= outshape.Ndim) {
                return false;
            }

            int ndim = outshape.Ndim;
            int length = 0;

            foreach (Shape inshape in inshapes) {
                if (inshape.Ndim != ndim) {
                    return false;
                }

                length += inshape[axis];

                for (int i = 0; i < ndim; i++) {
                    if (i == axis) {
                        continue;
                    }

                    if (inshape[i] != outshape[i]) {
                        return false;
                    }
                }
            }

            if (length != outshape[axis]) {
                return false;
            }

            return true;
        }
    }
}
