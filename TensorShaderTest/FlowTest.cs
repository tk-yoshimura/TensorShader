using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.VariableNode;

namespace TensorShaderTest {
    [TestClass]
    public class FlowTest {
        [TestMethod]
        public void MinimumTest() {
            {
                var v1 = new InputNode(Shape.Scalar);
                var v2 = v1.Save();

                Flow flow = Flow.FromInputs(v1);

                Tensor intensor = flow.InTensors[v1], outtensor = flow.OutTensors[v2];

                intensor.State = new float[] { 2 };
                outtensor.State = new float[] { 255 };

                flow.Execute();

                Assert.AreEqual(2, intensor.State[0]);
                Assert.AreEqual(2, outtensor.State[0]);
                Assert.AreEqual(2, flow.TensorCount);
                Assert.AreEqual(2, flow.BufferCount);
                Assert.AreEqual(2, flow.NodeCount);
            }

            {
                Tensor tensor = 1;

                var v1 = new InputNode(Shape.Scalar, tensor);
                v1.Update(v1);

                Flow flow = Flow.FromInputs(v1);

                flow.Execute();

                Assert.AreEqual(1, tensor.State[0]);
                Assert.AreEqual(1, flow.TensorCount);
                Assert.AreEqual(1, flow.BufferCount);
                Assert.AreEqual(2, flow.NodeCount);
            }
        }

        [TestMethod]
        public void DirectTest() {
            {
                var v1 = new InputNode(Shape.Scalar);
                var v2 = -v1;
                var v3 = -v2;
                var v4 = v3.Save();

                Flow flow = Flow.FromInputs(v1);

                Tensor intensor = flow.InTensors[v1], outtensor = flow.OutTensors[v4];

                intensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(2, outtensor.State[0]);
                Assert.AreEqual(4, flow.TensorCount);
                Assert.AreEqual(2, flow.BufferCount);
            }

            {
                Tensor tensor = Shape.Scalar;

                var v1 = new InputNode(Shape.Scalar, tensor);
                var v2 = -v1;
                var v3 = -v2;
                var v4 = v3.Save();
                var v5 = v2.Update(v1);

                Flow flow = Flow.FromInputs(v1);

                Tensor intensor = flow.InTensors[v1], outtensor1 = flow.OutTensors[v4], outtensor2 = flow.OutTensors[v5];

                intensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(2, outtensor1.State[0]);
                Assert.AreEqual(-2, outtensor2.State[0]);
                Assert.AreEqual(4, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);
            }

            {
                Tensor tensor = Shape.Scalar;

                var v1 = new InputNode(Shape.Scalar, tensor);
                var v2 = -v1;
                var v5 = v2.Update(v1);
                var v3 = -v2;
                var v4 = v3.Save();

                Flow flow = Flow.FromInputs(v1);

                Tensor intensor = flow.InTensors[v1], outtensor1 = flow.OutTensors[v4], outtensor2 = flow.OutTensors[v5];

                intensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(2, outtensor1.State[0]);
                Assert.AreEqual(-2, outtensor2.State[0]);
                Assert.AreEqual(4, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);
            }
        }

        [TestMethod]
        public void In1Out2Test() {
            var v1 = new InputNode(Shape.Scalar);
            var v2 = -v1;
            var v3 = -v2;
            var v4 = v3.Save();
            var v5 = v2.Save();

            {
                Flow flow = Flow.FromInputs(v1);

                Tensor intensor = flow.InTensors[v1], outtensor1 = flow.OutTensors[v4], outtensor2 = flow.OutTensors[v5];

                intensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(2, outtensor1.State[0]);
                Assert.AreEqual(-2, outtensor2.State[0]);
                Assert.AreEqual(5, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);
            }

            {
                Flow flow = Flow.FromOutputs(v4);

                Tensor intensor = flow.InTensors[v1], outtensor = flow.OutTensors[v4];

                intensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(2, outtensor.State[0]);
                Assert.AreEqual(4, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);

                CollectionAssert.AreEqual(new OutputNode[] { v4 }, flow.OutTensors.Keys.ToArray());
            }
        }

        [TestMethod]
        public void In2Out1Test() {
            var v1 = new InputNode(Shape.Scalar);
            var v2 = new InputNode(Shape.Scalar);
            var v3 = v1 + v2;
            v3 += v1;

            var v4 = -v3;
            var v5 = v4.Save();

            Flow flow = Flow.FromInputs(v1, v2);

            Tensor intensor1 = flow.InTensors[v1], intensor2 = flow.InTensors[v2], outtensor = flow.OutTensors[v5];

            intensor1.State = new float[] { 2 };
            intensor2.State = new float[] { 4 };

            flow.Execute();

            Assert.AreEqual(-8, outtensor.State[0]);
            Assert.AreEqual(6, flow.TensorCount);
            Assert.AreEqual(3, flow.BufferCount);
        }

        [TestMethod]
        public void ResidualConnectionTest() {
            {
                var v1 = new InputNode(Shape.Scalar); //x
                var v2 = Rcp(v1); // 1/x
                var v3 = Rcp(v2); // x
                var v4 = Rcp(v3); // 1/x
                var v5 = Rcp(v4); // x
                var v6 = v5 + v4;  // 1/x + x
                var v7 = v3 + v6;  // 1/x + 2x
                var v8 = v7 + v2;  // 2/x + 2x
                var v9 = v8 + v1;  // 2/x + 3x
                var v10 = v9.Save(); // 2/x + 3x
                var v11 = v7.Save(); // 1/x + 2x
                Flow flow = Flow.FromInputs(v1);

                Tensor intensor = flow.InTensors[v1], outtensor1 = flow.OutTensors[v10], outtensor2 = flow.OutTensors[v11];

                intensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(7, outtensor1.State[0]);
                Assert.AreEqual(4.5, outtensor2.State[0]);
                Assert.AreEqual(11, flow.TensorCount);
                Assert.AreEqual(5, flow.BufferCount);
            }

            {
                Tensor tensor = Shape.Scalar;

                var v1 = new InputNode(Shape.Scalar, tensor); //x
                var v2 = Rcp(v1); // 1/x
                var v3 = Rcp(v2); // x
                var v4 = Rcp(v3); // 1/x
                var v5 = Rcp(v4); // x
                var v6 = v5 + v4;  // 1/x + x
                var v7 = v3 + v6;  // 1/x + 2x
                var v8 = v7 + v2;  // 2/x + 2x
                var v9 = v8 + v1;  // 2/x + 3x
                var v10 = v9.Save(); // 2/x + 3x
                var v11 = v7.Save(); // 1/x + 2x
                _ = v5.Update(v1);

                Flow flow = Flow.FromInputs(v1);

                Tensor outtensor1 = flow.OutTensors[v10], outtensor2 = flow.OutTensors[v11];

                tensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(2, tensor.State[0]);
                Assert.AreEqual(7, outtensor1.State[0]);
                Assert.AreEqual(4.5, outtensor2.State[0]);
                Assert.AreEqual(11, flow.TensorCount);
                Assert.AreEqual(5, flow.BufferCount);
            }

            {
                Tensor tensor = Shape.Scalar;

                var v1 = new InputNode(Shape.Scalar, tensor); //x
                var v2 = Rcp(v1); // 1/x
                var v3 = Rcp(v2); // x
                var v4 = Rcp(v3); // 1/x
                var v5 = Rcp(v4); // x
                var v6 = v5 + v4;  // 1/x + x
                var v7 = v3 + v6;  // 1/x + 2x
                var v8 = v7 + v2;  // 2/x + 2x
                var v9 = v8 + v1;  // 2/x + 3x
                _ = v9.Save(); // 2/x + 3x
                var v11 = v7.Save(); // 1/x + 2x
                _ = v5.Update(v1);

                Flow flow = Flow.FromOutputs(v11);

                Tensor outtensor = flow.OutTensors[v11];

                tensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(2, tensor.State[0]);
                Assert.AreEqual(4.5, outtensor.State[0]);
                Assert.AreEqual(8, flow.TensorCount);
                Assert.AreEqual(4, flow.BufferCount);
            }

            {
                Tensor tensor = Shape.Scalar;

                var v1 = new InputNode(Shape.Scalar, tensor); //x
                var v2 = Rcp(v1); // 1/x
                var v3 = Rcp(v2); // x
                var v4 = Rcp(v3); // 1/x
                var v5 = Rcp(v4); // x
                var v6 = v5 + v4;  // 1/x + x
                var v7 = v3 + v6;  // 1/x + 2x
                var v8 = v7 + v2;  // 2/x + 2x
                var v9 = v8 + v1;  // 2/x + 3x
                _ = v9.Save(); // 2/x + 3x
                _ = v7.Save(); // 1/x + 2x
                var v12 = v5.Update(v1);

                Flow flow = Flow.FromOutputs(v12);

                tensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(2, tensor.State[0]);
                Assert.AreEqual(5, flow.TensorCount);
                Assert.AreEqual(2, flow.BufferCount);
            }
        }

        [TestMethod]
        public void Direct2Test() {
            {
                var v1 = new InputNode(Shape.Scalar);
                var v2 = v1 + 1;
                var v3 = v2 + 2;
                var v4 = v3.Save();

                Flow flow = Flow.FromInputs(v1);

                Tensor intensor = flow.InTensors[v1], outtensor = flow.OutTensors[v4];

                intensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(5, outtensor.State[0]);
                Assert.AreEqual(4, flow.TensorCount);
                Assert.AreEqual(2, flow.BufferCount);
            }

            {
                Tensor tensor = Shape.Scalar;

                var v1 = new InputNode(Shape.Scalar, tensor);
                var v2 = v1 + 1;
                var v3 = v2 + 2;
                var v4 = v3.Save();
                _ = v3.Update(v1);

                Flow flow = Flow.FromInputs(v1);

                Tensor outtensor = flow.OutTensors[v4];

                tensor.State = new float[] { 2 };

                flow.Execute();

                Assert.AreEqual(5, outtensor.State[0]);
                Assert.AreEqual(5, tensor.State[0]);
                Assert.AreEqual(4, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);
                Assert.AreNotEqual(tensor, outtensor);
            }
        }

        [TestMethod]
        public void CounterTest() {
            Tensor t = 1;

            var v1 = new InputNode(Shape.Scalar, t);
            var v2 = v1 + 1;
            var v3 = v2.Update(v1);

            Flow flow = Flow.FromInputs(v1);

            flow.Execute();
            flow.Execute();
            flow.Execute();
            flow.Execute();

            Assert.AreEqual(t, flow.InTensors[v1]);
            Assert.AreEqual(t, flow.OutTensors[v3]);

            Assert.AreEqual(5, t.State[0]);

            Assert.AreEqual(2, flow.TensorCount);
            Assert.AreEqual(2, flow.BufferCount);
        }

        [TestMethod]
        public void DoubleFlowTest() {
            Tensor t = 1;

            var v1 = new InputNode(Shape.Scalar, t);
            var v2 = v1 + 1;
            var v3 = v2.Update(v1);

            Flow flow1 = Flow.FromInputs(v1);
            Flow flow2 = Flow.FromInputs(v1);

            flow1.Execute();
            flow2.Execute();
            flow1.Execute();
            flow2.Execute();

            Assert.AreEqual(t, flow1.InTensors[v1]);
            Assert.AreEqual(t, flow1.OutTensors[v3]);

            Assert.AreEqual(t, flow2.InTensors[v1]);
            Assert.AreEqual(t, flow2.OutTensors[v3]);

            Assert.AreEqual(5, t.State[0]);

            Assert.AreEqual(2, flow1.TensorCount);
            Assert.AreEqual(2, flow1.BufferCount);

            Assert.AreEqual(2, flow2.TensorCount);
            Assert.AreEqual(2, flow2.BufferCount);
        }

        [TestMethod]
        public void CrossTest() {
            {
                Tensor t1 = 1;
                Tensor t2 = 3;

                var v1 = new InputNode(Shape.Scalar, t1);
                var v2 = v1 + 1;

                var v4 = new InputNode(Shape.Scalar, t2);
                var v5 = v4 + 1;

                var v3 = v2.Update(v4);
                var v6 = v5.Update(v1);

                Flow flow = Flow.FromInputs(v1, v4);

                flow.Execute();

                Assert.AreEqual(t1, flow.InTensors[v1]);
                Assert.AreEqual(t2, flow.OutTensors[v3]);

                Assert.AreEqual(t2, flow.InTensors[v4]);
                Assert.AreEqual(t1, flow.OutTensors[v6]);

                Assert.AreEqual(4, t1.State[0]);
                Assert.AreEqual(2, t2.State[0]);

                Assert.AreEqual(4, flow.TensorCount);
                Assert.AreEqual(4, flow.BufferCount);
            }

            {
                Tensor t1 = 1;
                Tensor t2 = 3;

                var v1 = new InputNode(Shape.Scalar, t1);
                var v2 = v1 + 1;

                var v4 = new InputNode(Shape.Scalar, t2);

                var v3 = v2.Update(v4);
                var v5 = v4.Update(v1);

                Flow flow = Flow.FromInputs(v1, v4);

                flow.Execute();

                Assert.AreEqual(t1, flow.InTensors[v1]);
                Assert.AreEqual(t2, flow.OutTensors[v3]);

                Assert.AreEqual(t2, flow.InTensors[v4]);
                Assert.AreEqual(t1, flow.OutTensors[v5]);

                Assert.AreEqual(3, t1.State[0]);
                Assert.AreEqual(2, t2.State[0]);

                Assert.AreEqual(3, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);
            }

            {
                Tensor t1 = 1;
                Tensor t2 = 3;

                var v1 = new InputNode(Shape.Scalar, t1);

                var v4 = new InputNode(Shape.Scalar, t2);
                var v5 = v4 + 1;

                var v3 = v1.Update(v4);
                var v6 = v5.Update(v1);

                Flow flow = Flow.FromInputs(v1, v4);

                flow.Execute();

                Assert.AreEqual(t1, flow.InTensors[v1]);
                Assert.AreEqual(t2, flow.OutTensors[v3]);

                Assert.AreEqual(t2, flow.InTensors[v4]);
                Assert.AreEqual(t1, flow.OutTensors[v6]);

                Assert.AreEqual(4, t1.State[0]);
                Assert.AreEqual(1, t2.State[0]);

                Assert.AreEqual(3, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);
            }
        }

        [TestMethod]
        public void ParallelTest() {
            Tensor t1 = 1;
            Tensor t2 = 3;
            Tensor t3 = 5;

            var v1 = new InputNode(Shape.Scalar, t1);
            var v2 = v1 + 1;

            var v4 = new InputNode(Shape.Scalar, t2);
            var v5 = v4 + 1;

            var v7 = new InputNode(Shape.Scalar, t3);
            var v8 = v7 + 1;

            var v3 = v2.Update(v4);
            var v6 = v5.Update(v7);
            var v9 = v8.Update(v1);

            Flow flow = Flow.FromInputs(v1, v4, v7);

            flow.Execute();

            Assert.AreEqual(t1, flow.InTensors[v1]);
            Assert.AreEqual(t2, flow.OutTensors[v3]);

            Assert.AreEqual(t2, flow.InTensors[v4]);
            Assert.AreEqual(t3, flow.OutTensors[v6]);

            Assert.AreEqual(t3, flow.InTensors[v7]);
            Assert.AreEqual(t1, flow.OutTensors[v9]);

            Assert.AreEqual(6, t1.State[0]);
            Assert.AreEqual(2, t2.State[0]);
            Assert.AreEqual(4, t3.State[0]);

            Assert.AreEqual(6, flow.TensorCount);
            Assert.AreEqual(6, flow.BufferCount);
        }

        [TestMethod]
        public void ReshapeTest() {
            float[] u = new float[12] { 7, 10, 2, 5, 11, 3, 8, 9, 4, 12, 1, 6, };
            Tensor t = (Shape.Map0D(3, 4), u);

            {
                var v1 = new InputNode(Shape.Map0D(3, 4), t);
                var v2 = v1 + 1;
                _ = Reshape(v2, Shape.Vector(12)).Save();

                Flow flow = Flow.FromInputs(v1);
                flow.Execute();

                Assert.AreEqual(4, flow.TensorCount);
                Assert.AreEqual(2, flow.BufferCount);
            }

            {
                var v1 = new InputNode(Shape.Map0D(3, 4), t);
                _ = Reshape(v1, Shape.Vector(12)).Save();
                _ = Reshape(v1, Shape.Map1D(2, 3, 2)).Save();

                Flow flow = Flow.FromInputs(v1);
                flow.Execute();

                Assert.AreEqual(5, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);
            }

            {
                var v1 = new InputNode(Shape.Map0D(3, 4), t);
                _ = Reshape(Reshape(v1, Shape.Vector(12)), Shape.Map0D(2, 6)).Save();
                _ = Reshape(v1, Shape.Map1D(2, 3, 2)).Save();

                Flow flow = Flow.FromInputs(v1);
                flow.Execute();

                Assert.AreEqual(6, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);
            }

            {
                var v1 = new InputNode(Shape.Map0D(3, 4), t);
                var v2 = v1 + 1;
                _ = Reshape(v2, Shape.Vector(12)).Save();
                _ = Reshape(v2, Shape.Map1D(2, 3, 2)).Save();

                Flow flow = Flow.FromInputs(v1);
                flow.Execute();

                Assert.AreEqual(6, flow.TensorCount);
                Assert.AreEqual(3, flow.BufferCount);
            }
        }
    }
}
