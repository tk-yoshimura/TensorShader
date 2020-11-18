using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.ComplexArithmetric {
    [TestClass]
    public class ComplexNormalizeTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() + 1).ToArray();
            float[] x2 = new float[length];

            {
                Tensor t1 = x1;
                Tensor o = Tensor.ComplexNormalize(t1);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 2; i++) {
                    Complex z = new Complex(x1[i * 2], x1[i * 2 + 1]);
                    Complex c = z / z.Magnitude;

                    Assert.AreEqual(c.Real, y[i * 2], 1e-6f, $"not equal {i}");
                    Assert.AreEqual(c.Imaginary, y[i * 2 + 1], 1e-6f, $"not equal {i}");
                }
            }

            {
                Tensor t2 = x2;
                Tensor o = Tensor.ComplexNormalize(t2);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 2; i++) {
                    Assert.AreEqual(0, y[i * 2], $"not equal {i}");
                    Assert.AreEqual(0, y[i * 2 + 1], $"not equal {i}");
                }
            }
        }
    }
}
