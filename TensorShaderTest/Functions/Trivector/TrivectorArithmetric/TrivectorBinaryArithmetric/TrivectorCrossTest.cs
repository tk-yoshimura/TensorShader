using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Numerics;
using TensorShader;

namespace TensorShaderTest.Functions.TrivectorArithmetric {
    [TestClass]
    public class TrivectorCrossTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 255;
            Random rd = new(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();
            float[] x2 = (new float[length]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();

            {
                Tensor t1 = x1;
                Tensor t2 = x2;
                Tensor o = Tensor.TrivectorCross(t1, t2);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 3; i++) {
                    Vector3 v = new(x1[i * 3], x1[i * 3 + 1], x1[i * 3 + 2]);
                    Vector3 u = new(x2[i * 3], x2[i * 3 + 1], x2[i * 3 + 2]);
                    Vector3 w = Vector3.Cross(v, u);

                    Assert.AreEqual(w.X, y[i * 3], 1e-6f, $"not equal {i * 3}");
                    Assert.AreEqual(w.Y, y[i * 3 + 1], 1e-6f, $"not equal {i * 3 + 1}");
                    Assert.AreEqual(w.Z, y[i * 3 + 2], 1e-6f, $"not equal {i * 3 + 2}");
                }
            }

            {
                Tensor t1 = x1;
                Tensor o = Tensor.TrivectorCross(t1, t1);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 3; i++) {
                    Vector3 v = new(x1[i * 3], x1[i * 3 + 1], x1[i * 3 + 2]);
                    Vector3 w = Vector3.Cross(v, v);

                    Assert.AreEqual(w.X, y[i * 3], 1e-6f, $"not equal {i * 3}");
                    Assert.AreEqual(w.Y, y[i * 3 + 1], 1e-6f, $"not equal {i * 3 + 1}");
                    Assert.AreEqual(w.Z, y[i * 3 + 2], 1e-6f, $"not equal {i * 3 + 2}");
                }
            }
        }
    }
}
