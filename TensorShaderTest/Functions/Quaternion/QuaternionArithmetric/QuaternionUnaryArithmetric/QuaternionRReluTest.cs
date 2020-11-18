using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.QuaternionArithmetric {
    [TestClass]
    public class QuaternionRReluTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();

            {
                Tensor t1 = x1;
                Tensor o = Tensor.QuaternionRRelu(t1);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 4; i++) {
                    Quaternion c = new Quaternion(x1[i * 4 + 1], x1[i * 4 + 2], x1[i * 4 + 3], Math.Max(0, x1[i * 4]));

                    Assert.AreEqual(c.W, y[i * 4], 1e-6f, $"not equal {i}");
                    Assert.AreEqual(c.X, y[i * 4 + 1], 1e-6f, $"not equal {i}");
                    Assert.AreEqual(c.Y, y[i * 4 + 2], 1e-6f, $"not equal {i}");
                    Assert.AreEqual(c.Z, y[i * 4 + 3], 1e-6f, $"not equal {i}");
                }
            }
        }
    }
}
