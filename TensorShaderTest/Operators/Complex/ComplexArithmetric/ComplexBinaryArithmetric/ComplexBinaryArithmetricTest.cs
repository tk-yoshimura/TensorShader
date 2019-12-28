using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexBinaryArithmetric;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexBinaryArithmetricTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1024 - 4; length <= i * 1024 + 4; length += 2) {
                    if (length < 1) continue;

                    float[] x = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] y = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape inshape = Shape.Vector(length);
                    Shape outshape = Shape.Vector(length);

                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(inshape, x);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(inshape, y);
                    OverflowCheckedTensor v3 = new OverflowCheckedTensor(outshape);

                    ComplexMul ope = new ComplexMul(inshape);

                    ope.Execute(v1, v2, v3);

                    CollectionAssert.AreEqual(x, v1.State);
                    CollectionAssert.AreEqual(y, v2.State);

                    v3.CheckOverflow();
                };
            }
        }
    }
}
