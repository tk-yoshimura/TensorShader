using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using TensorShader;
using TensorShader.Initializers;

namespace TensorShaderTest.Initializers {
    [TestClass]
    public class TrivectorNormalTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new(1234);

            const int batches = 1000, dims = 3, tests = 8;

            foreach (int num in new int[] { 32, 64, 128, 256 }) {
                List<float> variance_list = new();

                for (int i = 0; i < tests; i++) {
                    Tensor x = Tensor.NormalRandom(Shape.Map0D(num * dims, batches), random);
                    Tensor w = Shape.Kernel0D(num * dims / 3 * 4, dims);

                    var initializer = new TrivectorNormal(w, random);
                    initializer.Execute();

                    Tensor y = Tensor.TrivectorDense(x, w);

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
