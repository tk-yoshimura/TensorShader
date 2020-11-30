using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class ExpandChannelTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            foreach (int expands in new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30 }) {
                for (int i = 0; i < 4; i++) {
                    for (int length = i * 1024 - 2; length <= i * 1024 + 2; length++) {
                        if (length < 1)
                            continue;

                        float[] x = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                        OverflowCheckedTensor v1 = new OverflowCheckedTensor(Shape.Vector(length), x);
                        OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Vector(expands * length));

                        ExpandChannel ope = new ExpandChannel(v1.Shape, expands);

                        ope.Execute(v1, v2);

                        float[] y = v2.State.Value;

                        for (int j = 0; j < length; j++) {
                            for (int f = 0; f < expands; f++) {
                                Assert.AreEqual(x[j], y[f + expands * j], 1e-6f, $"length:{length}, idx:{j}");
                            }
                        }

                        Console.WriteLine($"pass: length:{length}, expands:{expands}");
                    }
                }
            }
        }
    }
}
