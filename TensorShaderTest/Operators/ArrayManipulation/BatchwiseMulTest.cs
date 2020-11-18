using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class BatchwiseMulTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            foreach (int batch in new int[] { 1, 2, 3, 4, 5 }) {
                for (int i = 0; i < 4; i++) {
                    for (int length = i * 1024 - 2; length <= i * 1024 + 2; length++) {
                        if (length < 1) continue;

                        foreach (int ch in new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }) {
                            float[] x1 = (new float[length * ch * batch]).Select((_) => (float)rd.NextDouble()).ToArray();
                            float[] x2 = (new float[batch]).Select((_) => (float)rd.NextDouble()).ToArray();

                            Shape shape = Shape.Map1D(ch, length, batch);

                            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape, x1);
                            OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Vector(shape.Batch), x2);
                            OverflowCheckedTensor v3 = new OverflowCheckedTensor(shape);

                            BatchwiseMul ope = new BatchwiseMul(shape);

                            ope.Execute(v1, v2, v3);

                            CollectionAssert.AreEqual(x1, v1.State.Value);
                            CollectionAssert.AreEqual(x2, v2.State.Value);

                            float[] y = v3.State.Value;

                            for (int j = 0; j < length * ch; j++) {
                                for (int k = 0; k < batch; k++)
                                    Assert.AreEqual(x1[j + length * ch * k] * x2[k], y[j + length * ch * k], 1e-6f, $"length:{length} ch:{ch} batch:{batch}, idx:{j},{k}");
                            }
                        }
                    };
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 65536, ch = 256, batch = 4;

            Shape shape = Shape.Map1D(ch, length, batch);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);
            OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Vector(batch));
            OverflowCheckedTensor v3 = new OverflowCheckedTensor(shape);

            BatchwiseMul ope = new BatchwiseMul(shape);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/batchwisemul.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(v1, v2, v3);

            Cuda.Profiler.Stop();
        }
    }
}
