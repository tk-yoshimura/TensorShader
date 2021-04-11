using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using TensorShader;
using TensorShader.Initializers;

namespace TensorShaderTest.Initializers {
    [TestClass]
    public class ComplexUniformTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new(1234);

            const int batches = 1000, dims = 2, tests = 8;
            float range = (float)Math.Sqrt(3);

            foreach (int num in new int[] { 32, 64, 128, 256 }) {
                List<float> variance_list = new();

                for (int i = 0; i < tests; i++) {
                    Tensor x = Tensor.UniformRandom(Shape.Map0D(num * dims, batches), random) * (2 * range) - range;
                    Tensor w = Shape.Kernel0D(num * dims, dims);

                    var initializer = new ComplexUniform(w, random);
                    initializer.Execute();

                    Tensor y = Tensor.ComplexDense(x, w);

                    float variance = Tensor.Average(y * y).State;

                    variance_list.Add(variance);
                }

                float mean_variance = variance_list.Average();

                Assert.AreEqual(1, mean_variance, 0.1f, $"{num}:{mean_variance}");

                Console.WriteLine($"{num}:{mean_variance}");
            }
        }
    }
}
