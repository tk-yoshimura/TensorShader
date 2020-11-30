using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class WhereTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (int i = 0; i < 32; i++) {
                for (int length = i * 1024 - 2; length <= i * 1024 + 2; length++) {
                    if (length < 1) continue;

                    float[] c = (new float[length]).Select((_) => rd.Next() % 2 == 0 ? 1f : 0f).ToArray();
                    float[] x1 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] x2 = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape shape = Shape.Vector(length);

                    OverflowCheckedTensor condition = new OverflowCheckedTensor(shape, c);
                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape, x1);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(shape, x2);
                    OverflowCheckedTensor vret = new OverflowCheckedTensor(shape);

                    Where ope = new Where(shape);

                    ope.Execute(condition, v1, v2, vret);

                    CollectionAssert.AreEqual(c, condition.State.Value);
                    CollectionAssert.AreEqual(x1, v1.State.Value);
                    CollectionAssert.AreEqual(x2, v2.State.Value);

                    float[] y = vret.State.Value;

                    for (int j = 0; j < length; j++) {
                        Assert.AreEqual(x1[j] * c[j] + x2[j] * (1 - c[j]), y[j], 1e-6f, $"length:{length}, idx:{j}");
                    }
                }
            }
        }
    }
}
