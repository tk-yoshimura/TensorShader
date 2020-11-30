using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorUnaryArithmetric;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorUnaryArithmetricTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1020 - 6; length <= i * 1020 + 6; length += 3) {
                    if (length < 1) continue;

                    float[] x = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape inshape = Shape.Vector(length);
                    Shape outshape = Shape.Vector(length);

                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(inshape, x);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(outshape);

                    TrivectorDecay ope = new TrivectorDecay(inshape);

                    ope.Execute(v1, v2);

                    CollectionAssert.AreEqual(x, v1.State.Value);

                    v2.CheckOverflow();
                };
            }
        }
    }
}
