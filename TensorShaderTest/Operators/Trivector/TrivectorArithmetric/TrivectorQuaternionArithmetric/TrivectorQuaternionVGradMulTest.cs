using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorQuaternionArithmetric;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorQuaternionVGradMulTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1020 - 6; length <= i * 1020 + 6; length += 3) {
                    if (length < 1) continue;

                    float[] v = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] q = (new float[length / 3 * 4]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape vecshape = Shape.Vector(length);
                    Shape quatshape = Shape.Vector(length / 3 * 4);

                    OverflowCheckedTensor v1 = new(vecshape, v);
                    OverflowCheckedTensor v2 = new(quatshape, q);
                    OverflowCheckedTensor v3 = new(vecshape);

                    TrivectorQuaternionMulVGrad ope = new(vecshape);

                    ope.Execute(v1, v2, v3);

                    CollectionAssert.AreEqual(v, v1.State.Value);
                    CollectionAssert.AreEqual(q, v2.State.Value);

                    v3.CheckOverflow();
                };
            }
        }
    }
}
