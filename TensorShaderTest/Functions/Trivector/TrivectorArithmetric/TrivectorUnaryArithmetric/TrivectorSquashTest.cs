using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Numerics;
using TensorShader;

namespace TensorShaderTest.Functions.TrivectorArithmetric {
    [TestClass]
    public class TrivectorSquashTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 255;
            Random rd = new(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();

            {
                Tensor t1 = x1;
                Tensor o = Tensor.TrivectorSquash(t1);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 3; i++) {
                    Vector3 v = new(x1[i * 3], x1[i * 3 + 1], x1[i * 3 + 2]);
                    Vector3 u = v / (v.Length() + 1);

                    Assert.AreEqual(u.X, y[i * 3], 1e-6f, $"not equal {i * 3}");
                    Assert.AreEqual(u.Y, y[i * 3 + 1], 1e-6f, $"not equal {i * 3 + 1}");
                    Assert.AreEqual(u.Z, y[i * 3 + 2], 1e-6f, $"not equal {i * 3 + 2}");
                }
            }
        }
    }
}
