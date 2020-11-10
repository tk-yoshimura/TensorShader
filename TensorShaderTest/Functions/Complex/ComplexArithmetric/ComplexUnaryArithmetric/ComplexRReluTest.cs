using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.ComplexArithmetric {
    [TestClass]
    public class ComplexRReluTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();

            {
                Tensor t1 = x1;
                Tensor o = Tensor.ComplexRRelu(t1);

                float[] y = o.State;

                for (int i = 0; i < y.Length / 2; i++) {
                    Complex c = new Complex(Math.Max(0, x1[i * 2]), x1[i * 2 + 1]);

                    Assert.AreEqual(c.Real, y[i * 2], 1e-6f, $"not equal {i}");
                    Assert.AreEqual(c.Imaginary, y[i * 2 + 1], 1e-6f, $"not equal {i}");
                }
            }
        }
    }
}
