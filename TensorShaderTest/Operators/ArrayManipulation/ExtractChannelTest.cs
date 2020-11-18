using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class ExtractChannelTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            foreach (int inch in new int[] { 11, 13 }) {
                foreach (int outch in new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }) {
                    foreach (int idx_ch in new int[] { 0, (inch - outch) / 2, inch - outch }) {
                        for (int i = 0; i < 4; i++) {
                            for (int length = i * 1024 - 2; length <= i * 1024 + 2; length++) {
                                if (length < 1)
                                    continue;

                                float[] x = (new float[length * inch]).Select((_) => (float)rd.NextDouble()).ToArray();

                                OverflowCheckedTensor v1 = new OverflowCheckedTensor(Shape.Map1D(inch, length), x);
                                OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map1D(outch, length));

                                ExtractChannel ope = new ExtractChannel(v1.Shape, idx_ch, v2.Shape);

                                ope.Execute(v1, v2);

                                float[] y = v2.State.Value;

                                for (int j = 0; j < length; j++) {
                                    for (int f = 0; f < outch; f++) {
                                        Assert.AreEqual(x[idx_ch + f + inch * j], y[f + outch * j], 1e-6f, $"length:{length}, idx:{j}");
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
