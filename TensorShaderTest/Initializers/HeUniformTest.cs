using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Initializers;

namespace TensorShaderTest.Initializers {
    [TestClass]
    public class HeUniformTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new Random(1234);

            const int batches = 1000, tests = 8;
            float range = (float)Math.Sqrt(3);

            foreach (int num in new int[] { 32, 64, 128, 256 }) {
                List<float> variance_list = new List<float>();

                for (int i = 0; i < tests; i++) {
                    Tensor x = Tensor.UniformRandom(Shape.Map0D(num, batches), random) * (2 * range) - range;
                    Tensor w = Shape.Kernel0D(num, 1);

                    var initializer = new HeUniform(w, random);
                    initializer.Execute();

                    Tensor y = Tensor.Relu(Tensor.Dense(x, w));

                    float variance = Tensor.Average(y * y).State[0];

                    variance_list.Add(variance);
                }

                float mean_variance = variance_list.Average();

                Assert.AreEqual(1, mean_variance, 0.1f, $"{num}:{mean_variance}");

                Console.WriteLine($"{num}:{mean_variance}");
            }
        }
    }
}
