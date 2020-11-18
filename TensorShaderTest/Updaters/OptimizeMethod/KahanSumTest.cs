using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class KahanSumTest {
        [TestMethod]
        public void ExecuteMethod() {
            const int loops = 1000;
            const double dx_val = 1.1;

            InputNode x = 0;
            InputNode dx = (float)dx_val;
            InputNode c = 0;

            (VariableNode new_x, VariableNode new_c) = TensorShader.Updaters.OptimizeMethod.OptimizeMethod.KahanSum(x, dx, c);

            new_x.Update(x);
            new_c.Update(c);

            Flow flow = Flow.FromInputs(x, dx, c);

            for (int i = 0; i < loops; i++) {
                flow.Execute();

                Console.WriteLine($"{i + 1}: {(float)x.State:F8}, {(float)c.State:F8}");
            }

            Assert.AreEqual(dx_val * loops, (float)x.State, 1e-5f);
        }
    }
}
