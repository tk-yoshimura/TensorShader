using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using TensorShader;
using TensorShaderCudaBackend;

namespace TensorShaderTest {
    [TestClass]
    public class TensorTest {
        [TestMethod]
        public void StateTest() {
            int channels = 12, batch = 4, length = channels * batch;

            Random random = new(1234);

            float[] v1 = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] v2 = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();

            {
                Tensor tensor = Shape.Map0D(channels, batch);

                Assert.AreEqual(length, tensor.Length);
                foreach (float v in tensor.State.Value) {
                    Assert.AreEqual(0f, v);
                }

                tensor.State = (tensor.Shape, v2);
                CollectionAssert.AreEqual(v2, tensor.State.Value);
            }

            {
                Tensor tensor = (Shape.Map0D(channels, batch), v1);

                Assert.AreEqual(length, tensor.Length);
                CollectionAssert.AreEqual(v1, tensor.State.Value);

                tensor.State = (tensor.Shape, v2);
                CollectionAssert.AreEqual(v2, tensor.State.Value);
            }

            {
                Tensor tensor = new OverflowCheckedTensor(Shape.Map0D(channels, batch));

                Assert.AreEqual(length, tensor.Length);
                foreach (float v in tensor.State.Value.Skip(1)) {
                    Assert.AreEqual(0f, v);
                }

                tensor.State = (tensor.Shape, v2);
                CollectionAssert.AreEqual(v2, tensor.State.Value);
            }

            {
                Tensor tensor = new OverflowCheckedTensor(Shape.Map0D(channels, batch), v1);

                Assert.AreEqual(length, tensor.Length);
                CollectionAssert.AreEqual(v1, tensor.State.Value);

                tensor.State = (tensor.Shape, v2);
                CollectionAssert.AreEqual(v2, tensor.State.Value);
            }
        }

        [TestMethod]
        public void ClearTest() {
            int channels = 12, batch = 4, length = channels * batch;

            Random random = new(1234);

            float[] v1 = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();

            {
                Tensor tensor = Shape.Map0D(channels, batch);

                tensor.Clear(1);

                Assert.AreEqual(length, tensor.Length);
                foreach (float v in tensor.State.Value) {
                    Assert.AreEqual(1f, v);
                }
            }

            {
                Tensor tensor = (Shape.Map0D(channels, batch), v1);

                tensor.Clear(1);

                Assert.AreEqual(length, tensor.Length);
                foreach (float v in tensor.State.Value) {
                    Assert.AreEqual(1f, v);
                }
            }

            {
                Tensor tensor = new OverflowCheckedTensor(Shape.Map0D(channels, batch));

                tensor.Clear(1);

                Assert.AreEqual(length, tensor.Length);
                foreach (float v in tensor.State.Value) {
                    Assert.AreEqual(1f, v);
                }
            }

            {
                Tensor tensor = new OverflowCheckedTensor(Shape.Map0D(channels, batch), v1);

                tensor.Clear(1);

                Assert.AreEqual(length, tensor.Length);
                foreach (float v in tensor.State.Value) {
                    Assert.AreEqual(1f, v);
                }
            }
        }

        [TestMethod]
        public void ZerosetTest() {
            int channels = 12, batch = 4, length = channels * batch;

            Random random = new(1234);

            float[] v1 = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();

            {
                Tensor tensor = (Shape.Map0D(channels, batch), v1);

                tensor.Zeroset();

                Assert.AreEqual(length, tensor.Length);
                foreach (float v in tensor.State.Value) {
                    Assert.AreEqual(0f, v);
                }
            }

            {
                Tensor tensor = new OverflowCheckedTensor(Shape.Map0D(channels, batch), v1);

                tensor.Zeroset();

                Assert.AreEqual(length, tensor.Length);
                foreach (float v in tensor.State.Value) {
                    Assert.AreEqual(0f, v);
                }
            }
        }

        [TestMethod]
        public void CopyToTest() {
            int channels = 12, batch = 4, length = channels * batch;

            Random random = new(1234);

            float[] v1 = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();

            {
                Tensor tensor1 = (Shape.Map0D(channels, batch), v1);
                Tensor tensor2 = Shape.Map0D(channels, batch);

                tensor1.CopyTo(tensor2);

                Assert.AreEqual(length, tensor2.Length);
                CollectionAssert.AreEqual(v1, tensor2.State.Value);
            }

            {
                Tensor tensor1 = (Shape.Map0D(channels, batch), v1);
                Tensor tensor2 = new OverflowCheckedTensor(Shape.Map0D(channels, batch));

                tensor1.CopyTo(tensor2);

                Assert.AreEqual(length, tensor2.Length);
                CollectionAssert.AreEqual(v1, tensor2.State.Value);
            }

            {
                Tensor tensor1 = new OverflowCheckedTensor(Shape.Map0D(channels, batch), v1);
                Tensor tensor2 = Shape.Map0D(channels, batch);

                tensor1.CopyTo(tensor2);

                Assert.AreEqual(length, tensor2.Length);
                CollectionAssert.AreEqual(v1, tensor2.State.Value);
            }

            {
                Tensor tensor1 = new OverflowCheckedTensor(Shape.Map0D(channels, batch), v1);
                Tensor tensor2 = new OverflowCheckedTensor(Shape.Map0D(channels, batch));

                tensor1.CopyTo(tensor2);

                Assert.AreEqual(length, tensor2.Length);
                CollectionAssert.AreEqual(v1, tensor2.State.Value);
            }
        }

        [TestMethod]
        public void RegionCopyToTest() {
            int ch1 = 2, ch2 = 3, batch = 4, length1 = ch1 * batch, length2 = ch2 * batch;

            Random random = new(1234);

            float[] v1 = (new float[length1]).Select((_) => (float)random.NextDouble()).ToArray();
            float[] v2 = (new float[length2]).Select((_) => (float)random.NextDouble()).ToArray();

            for (int src_index = 0; src_index < length1; src_index++) {
                for (int dst_index = 0; dst_index < length2; dst_index++) {
                    int max_count = Math.Min(length1 - src_index, length2 - dst_index);

                    for (int count = 0; count <= max_count; count++) {
                        float[] v = (float[])v2.Clone();
                        for (int i = 0; i < count; i++) {
                            v[dst_index + i] = v1[src_index + i];
                        }

                        {
                            Tensor tensor1 = (Shape.Map0D(ch1, batch), v1);
                            Tensor tensor2 = (Shape.Map0D(ch2, batch), v2);

                            tensor1.RegionCopyTo(tensor2, (uint)src_index, (uint)dst_index, (uint)count);

                            CollectionAssert.AreEqual(v, tensor2.State.Value);
                        }

                        {
                            Tensor tensor1 = (Shape.Map0D(ch1, batch), v1);
                            Tensor tensor2 = new OverflowCheckedTensor(Shape.Map0D(ch2, batch), v2);

                            tensor1.RegionCopyTo(tensor2, (uint)src_index, (uint)dst_index, (uint)count);

                            CollectionAssert.AreEqual(v, tensor2.State.Value);
                        }

                        {
                            Tensor tensor1 = new OverflowCheckedTensor(Shape.Map0D(ch1, batch), v1);
                            Tensor tensor2 = (Shape.Map0D(ch2, batch), v2);

                            tensor1.RegionCopyTo(tensor2, (uint)src_index, (uint)dst_index, (uint)count);

                            CollectionAssert.AreEqual(v, tensor2.State.Value);
                        }

                        {
                            Tensor tensor1 = new OverflowCheckedTensor(Shape.Map0D(ch1, batch), v1);
                            Tensor tensor2 = new OverflowCheckedTensor(Shape.Map0D(ch2, batch), v2);

                            tensor1.RegionCopyTo(tensor2, (uint)src_index, (uint)dst_index, (uint)count);

                            CollectionAssert.AreEqual(v, tensor2.State.Value);
                        }

                    }

                    for (int count = max_count + 1; count <= max_count + 2; count++) {
                        {
                            Tensor tensor1 = (Shape.Map0D(ch1, batch), v1);
                            Tensor tensor2 = (Shape.Map0D(ch2, batch), v2);

                            Assert.ThrowsException<ArgumentOutOfRangeException>(
                                () => { tensor1.RegionCopyTo(tensor2, (uint)src_index, (uint)dst_index, (uint)count); }
                            );
                        }

                        {
                            Tensor tensor1 = (Shape.Map0D(ch1, batch), v1);
                            Tensor tensor2 = new OverflowCheckedTensor(Shape.Map0D(ch2, batch), v2);

                            Assert.ThrowsException<ArgumentOutOfRangeException>(
                                () => { tensor1.RegionCopyTo(tensor2, (uint)src_index, (uint)dst_index, (uint)count); }
                            );
                        }

                        {
                            Tensor tensor1 = new OverflowCheckedTensor(Shape.Map0D(ch1, batch), v1);
                            Tensor tensor2 = (Shape.Map0D(ch2, batch), v2);

                            Assert.ThrowsException<ArgumentOutOfRangeException>(
                                () => { tensor1.RegionCopyTo(tensor2, (uint)src_index, (uint)dst_index, (uint)count); }
                            );
                        }

                        {
                            Tensor tensor1 = new OverflowCheckedTensor(Shape.Map0D(ch1, batch), v1);
                            Tensor tensor2 = new OverflowCheckedTensor(Shape.Map0D(ch2, batch), v2);

                            Assert.ThrowsException<ArgumentOutOfRangeException>(
                                () => { tensor1.RegionCopyTo(tensor2, (uint)src_index, (uint)dst_index, (uint)count); }
                            );
                        }

                    }
                }
            }
        }

        [TestMethod]
        public void CopyTest() {
            int channels = 12, batch = 4, length = channels * batch;

            Random random = new(1234);

            float[] v1 = (new float[length]).Select((_) => (float)random.NextDouble()).ToArray();

            {
                Tensor tensor1 = (Shape.Map0D(channels, batch), v1);
                Tensor tensor2 = tensor1.Copy();

                Assert.AreEqual(length, tensor2.Length);
                CollectionAssert.AreEqual(v1, tensor2.State.Value);
            }

            {
                Tensor tensor1 = new OverflowCheckedTensor(Shape.Map0D(channels, batch), v1);
                Tensor tensor2 = tensor1.Copy();

                Assert.IsTrue(tensor2 is OverflowCheckedTensor);
                Assert.AreEqual(length, tensor2.Length);
                CollectionAssert.AreEqual(v1, tensor2.State.Value);
            }
        }

        [TestMethod]
        public void OverflowTest() {
            OverflowCheckedTensor tensor = new(Shape.Vector(12));

            Operator overflow = new OverflowOperator(tensor.Shape);

            NdimArray<float> vs = tensor.State;

            Assert.ThrowsException<AccessViolationException>(() => {
                overflow.Execute(tensor);

                vs = tensor.State;
            });

            Console.WriteLine("pass");
        }

        internal class OverflowOperator : Operator {
            /// <summary>形状</summary>
            public Shape Shape { private set; get; }

            /// <summary>コンストラクタ</summary>
            public OverflowOperator(Shape shape) {
                this.arguments = new List<(ArgumentType type, Shape shape)>{
                    (ArgumentType.Out, shape),
                };

                this.Shape = shape;
            }

            /// <summary>操作を実行</summary>
            public override void Execute(params Tensor[] tensors) {
                CheckArgumentShapes(tensors);

                Tensor outmap = tensors[0];

                string code = @"
                __global__ void add(float *x, int length) {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    x[length] = 1;
                }";

                Kernel kernel = new(code, "add");

                kernel.Execute((uint)outmap.Length, dynamic_shared_memory_bytes: 0, stream: null, outmap.Buffer, outmap.Length);
            }

            /// <summary>操作を実行</summary>
            public void Execute(Tensor outmap) {
                Execute(new Tensor[] { outmap });
            }
        }
    }
}
