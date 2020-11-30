using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class InsertChannelTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            foreach (int outch in new int[] { 11, 12 }) {
                foreach (int inch in new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }) {
                    foreach (int idx_ch in new int[] { 0, (outch - inch) / 2, outch - inch }) {
                        for (int i = 0; i < 4; i++) {
                            for (int length = i * 1024 - 2; length <= i * 1024 + 2; length++) {
                                if (length < 1)
                                    continue;

                                float[] x = (new float[length * inch]).Select((_) => (float)rd.NextDouble()).ToArray();

                                OverflowCheckedTensor v1 = new OverflowCheckedTensor(Shape.Map1D(inch, length), x);
                                OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map1D(outch, length));

                                InsertChannel ope = new InsertChannel(v1.Shape, idx_ch, v2.Shape);

                                ope.Execute(v1, v2);

                                float[] y = v2.State.Value;

                                for (int j = 0; j < length; j++) {
                                    for (int f = 0; f < idx_ch; f++) {
                                        Assert.AreEqual(0f, y[f + outch * j], 1e-6f, $"length:{length}, idx:{j}");
                                    }
                                    for (int f = 0; f < inch; f++) {
                                        Assert.AreEqual(x[f + inch * j], y[idx_ch + f + outch * j], 1e-6f, $"length:{length}, idx:{j}");
                                    }
                                    for (int f = 0; f < outch - idx_ch - inch; f++) {
                                        Assert.AreEqual(0f, y[idx_ch + inch + f + outch * j], 1e-6f, $"length:{length}, idx:{j}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
