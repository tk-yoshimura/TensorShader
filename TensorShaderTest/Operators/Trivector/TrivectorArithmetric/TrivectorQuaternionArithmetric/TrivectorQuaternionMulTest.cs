using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorQuaternionArithmetric;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorQuaternionMulTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1020 - 6; length <= i * 1020 + 6; length += 3) {
                    if (length < 1) continue;

                    float[] v = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] q = (new float[length / 3 * 4]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape vecshape = Shape.Vector(length);
                    Shape quatshape = Shape.Vector(length / 3 * 4);

                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(vecshape, v);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(quatshape, q);
                    OverflowCheckedTensor v3 = new OverflowCheckedTensor(vecshape);

                    TrivectorQuaternionMul ope = new TrivectorQuaternionMul(vecshape);

                    ope.Execute(v1, v2, v3);

                    CollectionAssert.AreEqual(v, v1.State);
                    CollectionAssert.AreEqual(q, v2.State);

                    v3.CheckOverflow();
                };
            }
        }
    }
}
