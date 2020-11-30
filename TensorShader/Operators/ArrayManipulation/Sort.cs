using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>ソート</summary>
    internal class Sort : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>軸長さ</summary>
        public int AxisLength { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Sort(Shape shape, int axis) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            this.Shape = shape;
            this.Stride = shape.Stride(axis);
            this.AxisLength = shape[axis];
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor intensor = tensors[0], outtensor = tensors[1];

            TensorShaderCudaBackend.ArrayManipulation.Sort((uint)Stride, (uint)AxisLength, (uint)(outtensor.Length / (Stride * AxisLength)), intensor.Buffer, outtensor.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor intensor, Tensor outtensor) {
            Execute(new Tensor[] { intensor, outtensor });
        }

        /// <summary>CUDAコードと等価なC#コード</summary>
        /// <remarks>イントロンコード</remarks>
        public static void ParallelCombSort(int[] x, int threads) {
            int length = x.Length;

            for (int h = (length + 1) / 2; h >= 2; h = h * 4 / 5) {
                Parallel.For(0, threads, (tidx) => {
                    for (int j = tidx; ; j += threads) {
                        int i = j % h + (j / h) * 2 * h;
                        if (i + h >= length) break;

                        if (x[i] > x[i + h]) {
                            int temp = x[i]; x[i] = x[i + h]; x[i + h] = temp;
                        }
                    }
                });

                Parallel.For(0, threads, (tidx) => {
                    for (int j = tidx; ; j += threads) {
                        int i = j % h + (j / h) * 2 * h + h;
                        if (i + h >= length) break;

                        if (x[i] > x[i + h]) {
                            int temp = x[i]; x[i] = x[i + h]; x[i + h] = temp;
                        }
                    }
                });
            }

            bool is_swaped = true;
            while (is_swaped) {
                is_swaped = false;

                Parallel.For(0, threads, (tidx) => {
                    for (int j = tidx; ; j += threads) {
                        int i = j * 2;
                        if (i + 1 >= length) break;

                        if (x[i] > x[i + 1]) {
                            int temp = x[i]; x[i] = x[i + 1]; x[i + 1] = temp;
                            is_swaped = true;
                        }
                    }
                });

                Parallel.For(0, threads, (tidx) => {
                    for (int j = tidx; ; j += threads) {
                        int i = j * 2 + 1;
                        if (i + 1 >= length) break;

                        if (x[i] > x[i + 1]) {
                            int temp = x[i]; x[i] = x[i + 1]; x[i + 1] = temp;
                            is_swaped = true;
                        }
                    }
                });
            }
        }
    }
}
