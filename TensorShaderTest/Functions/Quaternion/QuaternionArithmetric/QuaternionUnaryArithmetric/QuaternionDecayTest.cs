using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Numerics;
using TensorShader;

namespace TensorShaderTest.Functions.QuaternionArithmetric {
    [TestClass]
    public class QuaternionDecayTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() + 1).ToArray();

            {
                Tensor t1 = x1;
                Tensor o = Tensor.QuaternionDecay(t1);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 4; i++) {
                    Quaternion z = new(x1[i * 4 + 1], x1[i * 4 + 2], x1[i * 4 + 3], x1[i * 4]);
                    Quaternion q = z * (z.Length() * z.Length() / (z.Length() * z.Length() + 1));

                    Assert.AreEqual(q.X, y[i * 4 + 1], 1e-6f, $"not equal {i * 4 + 1}");
                    Assert.AreEqual(q.Y, y[i * 4 + 2], 1e-6f, $"not equal {i * 4 + 2}");
                    Assert.AreEqual(q.Z, y[i * 4 + 3], 1e-6f, $"not equal {i * 4 + 3}");
                    Assert.AreEqual(q.W, y[i * 4], 1e-6f, $"not equal {i * 4}");
                }
            }
        }
    }
}
