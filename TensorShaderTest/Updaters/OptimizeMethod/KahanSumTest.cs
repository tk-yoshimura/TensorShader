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

            InputNode x = (Shape.Scalar, new float[] { 0 });
            InputNode dx = (Shape.Scalar, new float[] { (float)dx_val });
            InputNode c = (Shape.Scalar, new float[] { 0 });

            (VariableNode new_x, VariableNode new_c) = TensorShader.Updaters.OptimizeMethod.OptimizeMethod.KahanSum(x, dx, c);

            new_x.Update(x);
            new_c.Update(c);

            Flow flow = Flow.FromInputs(x, dx, c);

            for (int i = 0; i < loops; i++) {
                flow.Execute();

                Console.WriteLine($"{i + 1}: {x.State[0]:F8}, {c.State[0]:F8}");
            }

            Assert.AreEqual(dx_val * loops, x.State[0], 1e-5f);
        }
    }
}
