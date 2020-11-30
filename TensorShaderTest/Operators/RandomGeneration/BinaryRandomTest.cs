using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.RandomGeneration;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.RandomGeneration {
    [TestClass]
    public class BinaryRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            int length = 65535 * 33;

            Shape shape = Shape.Vector(length);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);

            for (decimal prob = 0; prob <= 1; prob += 0.05m) {
                BinaryRandom ope = new BinaryRandom(shape, new Random(1234), (float)prob);

                ope.Execute(v1);

                float[] y = v1.State.Value;

                Assert.AreEqual((float)prob, y.Average(), 1e-3f);
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 65535 * 15;

            Shape shape = Shape.Vector(length);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);

            BinaryRandom ope = new BinaryRandom(shape, new Random(1234), 0.25f);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/binary_random.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(v1);

            Cuda.Profiler.Stop();
        }
    }
}
