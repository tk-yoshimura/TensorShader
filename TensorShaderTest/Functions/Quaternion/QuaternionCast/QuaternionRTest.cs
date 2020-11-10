using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.QuaternionArithmetric {
    [TestClass]
    public class QuaternionRealTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();

            {
                Tensor t1 = (Shape.Vector(length), x1);
                Tensor o = Tensor.QuaternionR(t1);

                float[] y = o.State;

                for (int i = 0; i < y.Length / 4; i++) {
                    Quaternion q = new Quaternion(x1[i * 4 + 1], x1[i * 4 + 2], x1[i * 4 + 3], x1[i * 4]);

                    Assert.AreEqual(q.W, y[i], 1e-6f, $"not equal {i}");
                }
            }
        }
    }
}
