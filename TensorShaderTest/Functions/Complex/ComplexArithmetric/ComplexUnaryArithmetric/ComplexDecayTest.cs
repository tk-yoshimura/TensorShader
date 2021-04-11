using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Numerics;
using TensorShader;

namespace TensorShaderTest.Functions.ComplexArithmetric {
    [TestClass]
    public class ComplexDecayTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() + 1).ToArray();

            {
                Tensor t1 = x1;
                Tensor o = Tensor.ComplexDecay(t1);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 2; i++) {
                    Complex z = new(x1[i * 2], x1[i * 2 + 1]);
                    Complex c = z * (z.Magnitude * z.Magnitude / (z.Magnitude * z.Magnitude + 1));

                    Assert.AreEqual(c.Real, y[i * 2], 1e-6f, $"not equal {i}");
                    Assert.AreEqual(c.Imaginary, y[i * 2 + 1], 1e-6f, $"not equal {i}");
                }
            }
        }
    }
}
