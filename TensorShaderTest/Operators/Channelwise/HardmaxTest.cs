using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Channelwise;

namespace TensorShaderTest.Operators.Channelwise {
    [TestClass]
    public class HardmaxTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new(1234);

            foreach (int indexes in new int[] { 1, 2, 3, 4, 8, 16, 32 }) {
                for (int channels = 1; channels <= 1024; channels *= 2) {
                    float[] x1 = (new float[channels * indexes]).Select((_) => (float)rd.NextDouble()).ToArray();

                    OverflowCheckedTensor v1 = new(Shape.Map0D(channels, indexes), x1);
                    OverflowCheckedTensor v2 = new(Shape.Map0D(channels, indexes));

                    Hardmax ope = new(channels, Shape.Map0D(channels, indexes));

                    ope.Execute(v1, v2);

                    CollectionAssert.AreEqual(x1, v1.State.Value);

                    float[] y = v2.State.Value;

                    for (int j = 0; j < indexes; j++) {
                        double max_v = 0;
                        int max_v_i = 0;

                        for (int k = 0; k < channels; k++) {
                            double v = x1[k + j * channels];

                            if (v > max_v) {
                                max_v = v;
                                max_v_i = k;
                            }
                        }

                        for (int k = 0; k < channels; k++) {
                            Assert.AreEqual(max_v_i == k ? 1 : 0, y[k + j * channels], 1e-6f, $"indexes:{indexes},channels:{channels} idx:{j}");
                        }
                    }
                }
            }
        }
    }
}
