using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Channelwise;

namespace TensorShaderTest.Operators.Channelwise {
    [TestClass]
    public class SoftmaxTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            foreach (int indexes in new int[] { 1, 2, 3, 4, 8, 16, 32 }) {
                for (int channels = 1; channels <= 1024; channels *= 2) {
                    float[] x1 = (new float[channels * indexes]).Select((_) => (float)rd.NextDouble()).ToArray();

                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(Shape.Map0D(channels, indexes), x1);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map0D(channels, indexes));

                    Softmax ope = new Softmax(channels, Shape.Map0D(channels, indexes));

                    ope.Execute(v1, v2);

                    CollectionAssert.AreEqual(x1, v1.State.Value);

                    float[] y = v2.State.Value;

                    for (int j = 0; j < indexes; j++) {
                        double sum_exp_v = 0;

                        for (int k = 0; k < channels; k++) {
                            double v = x1[k + j * channels];
                            double exp_v = Math.Exp(v);

                            sum_exp_v += exp_v;
                        }

                        for (int k = 0; k < channels; k++) {
                            double v = x1[k + j * channels];
                            double exp_v = Math.Exp(v);

                            Assert.AreEqual(exp_v / sum_exp_v, y[k + j * channels], 1e-6f, $"indexes:{indexes},channels:{channels} idx:{j}");
                        }
                    }
                }
            }
        }
    }
}
