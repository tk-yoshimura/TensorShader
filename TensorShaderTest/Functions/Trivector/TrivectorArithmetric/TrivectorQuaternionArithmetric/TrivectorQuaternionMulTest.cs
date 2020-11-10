using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Functions.TrivectorArithmetric;

namespace TensorShaderTest.Functions.TrivectorArithmetric {
    [TestClass]
    public class TrivectorQuaternionMulTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 255;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();
            float[] x2 = (new float[length / 3 * 4]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();

            {
                Tensor t1 = x1;
                Tensor t2 = (Shape.Vector(length / 3 * 4), x2);
                Tensor o = Tensor.TrivectorQuaternionMul(t1, t2);

                float[] y = o.State;

                for (int i = 0; i < y.Length / 3; i++) {
                    Quaternion v = new Quaternion(new Vector3(x1[i * 3], x1[i * 3 + 1], x1[i * 3 + 2]), 0);
                    Quaternion q = new Quaternion(x2[i * 4 + 1], x2[i * 4 + 2], x2[i * 4 + 3], x2[i * 4]);
                    Quaternion p = q * v * Quaternion.Conjugate(q);

                    Assert.AreEqual(p.X, y[i * 3], 1e-6f, $"not equal {i * 3}");
                    Assert.AreEqual(p.Y, y[i * 3 + 1], 1e-6f, $"not equal {i * 3 + 1}");
                    Assert.AreEqual(p.Z, y[i * 3 + 2], 1e-6f, $"not equal {i * 3 + 2}");
                }
            }

            Function function = new TrivectorQuaternionMul();

            {
                Tensor t1 = x1;
                Tensor t2 = (Shape.Vector(length / 3 * 4), x2);
                Tensor o = Shape.Vector(length);

                function.Execute(new Tensor[] { t1, t2 }, new Tensor[] { o });

                float[] y = o.State;

                for (int i = 0; i < y.Length / 3; i++) {
                    Quaternion v = new Quaternion(new Vector3(x1[i * 3], x1[i * 3 + 1], x1[i * 3 + 2]), 0);
                    Quaternion q = new Quaternion(x2[i * 4 + 1], x2[i * 4 + 2], x2[i * 4 + 3], x2[i * 4]);
                    Quaternion p = q * v * Quaternion.Conjugate(q);

                    Assert.AreEqual(p.X, y[i * 3], 1e-6f, $"not equal {i * 3}");
                    Assert.AreEqual(p.Y, y[i * 3 + 1], 1e-6f, $"not equal {i * 3 + 1}");
                    Assert.AreEqual(p.Z, y[i * 3 + 2], 1e-6f, $"not equal {i * 3 + 2}");
                }
            }
        }
    }
}
