using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexUnaryArithmetric;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexUnaryArithmetricTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1024 - 4; length <= i * 1024 + 4; length += 2) {
                    if (length < 1) continue;

                    float[] x = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape inshape = Shape.Vector(length);
                    Shape outshape = Shape.Vector(length);

                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(inshape, x);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(outshape);

                    ComplexConjugate ope = new ComplexConjugate(inshape);

                    ope.Execute(v1, v2);

                    CollectionAssert.AreEqual(x, v1.State);

                    v2.CheckOverflow();
                };
            }
        }
    }
}
