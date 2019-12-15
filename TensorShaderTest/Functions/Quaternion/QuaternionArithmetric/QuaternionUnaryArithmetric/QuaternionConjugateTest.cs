using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.QuaternionArithmetric {
    [TestClass]
    public class QuaternionConjugateTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();

            {
                Tensor t1 = new Tensor(Shape.Vector(length), x1);
                Tensor o = Tensor.QuaternionConjugate(t1);

                float[] y = o.State;

                for (int i = 0; i < y.Length / 4; i++) {
                    Quaternion z = new Quaternion(x1[i * 4 + 1], x1[i * 4 + 2], x1[i * 4 + 3], x1[i * 4]);
                    Quaternion q = Quaternion.Conjugate(z);

                    Assert.AreEqual(q.X, y[i * 4 + 1], 1e-6f, $"not equal {i * 4 + 1}");
                    Assert.AreEqual(q.Y, y[i * 4 + 2], 1e-6f, $"not equal {i * 4 + 2}");
                    Assert.AreEqual(q.Z, y[i * 4 + 3], 1e-6f, $"not equal {i * 4 + 3}");
                    Assert.AreEqual(q.W, y[i * 4], 1e-6f, $"not equal {i * 4}");
                }
            }
        }
    }
}
