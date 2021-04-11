using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.QuaternionBinaryArithmetric;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionBinaryArithmetricTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1024 - 4; length <= i * 1024 + 4; length += 4) {
                    if (length < 1) continue;

                    float[] x = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] y = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape inshape = Shape.Vector(length);
                    Shape outshape = Shape.Vector(length);

                    OverflowCheckedTensor v1 = new(inshape, x);
                    OverflowCheckedTensor v2 = new(inshape, y);
                    OverflowCheckedTensor v3 = new(outshape);

                    QuaternionMul ope = new(inshape);

                    ope.Execute(v1, v2, v3);

                    CollectionAssert.AreEqual(x, v1.State.Value);
                    CollectionAssert.AreEqual(y, v2.State.Value);

                    v3.CheckOverflow();
                };
            }
        }
    }
}
