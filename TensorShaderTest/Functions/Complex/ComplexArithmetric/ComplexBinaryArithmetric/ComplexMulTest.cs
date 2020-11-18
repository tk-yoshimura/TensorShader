using System;
using System.Linq;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.ComplexArithmetric {
    [TestClass]
    public class ComplexMulTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();
            float[] x2 = (new float[length]).Select((_) => (float)rd.NextDouble() - 0.5f).ToArray();

            {
                Tensor t1 = x1;
                Tensor t2 = x2;
                Tensor o = Tensor.ComplexMul(t1, t2);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 2; i++) {
                    Complex a = new Complex(x1[i * 2], x1[i * 2 + 1]);
                    Complex b = new Complex(x2[i * 2], x2[i * 2 + 1]);
                    Complex c = a * b;

                    Assert.AreEqual(c.Real, y[i * 2], 1e-6f, $"not equal {i}");
                    Assert.AreEqual(c.Imaginary, y[i * 2 + 1], 1e-6f, $"not equal {i}");
                }
            }

            {
                Tensor t1 = x1;
                Tensor o = Tensor.ComplexMul(t1, t1);

                float[] y = o.State.Value;

                for (int i = 0; i < y.Length / 2; i++) {
                    Complex a = new Complex(x1[i * 2], x1[i * 2 + 1]);
                    Complex c = a * a;

                    Assert.AreEqual(c.Real, y[i * 2], 1e-6f, $"not equal {i}");
                    Assert.AreEqual(c.Imaginary, y[i * 2 + 1], 1e-6f, $"not equal {i}");
                }
            }
        }
    }
}
