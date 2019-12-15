using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Indexer;

namespace TensorShaderTest.Operators.Indexer {
    [TestClass]
    public class ArgMinTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            foreach (int batch in new int[] { 1, 2, 3, 4, 8, 16, 32 }) {
                for (int channels = 1; channels <= 1024; channels *= 2) {
                    float[] x1 = (new float[channels * batch]).Select((_) => (float)rd.NextDouble()).ToArray();

                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(Shape.Map0D(channels, batch), x1);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Vector(batch));

                    ArgMin ope = new ArgMin(channels, batch);

                    ope.Execute(v1, v2);

                    CollectionAssert.AreEqual(x1, v1.State);

                    float[] y = v2.State;

                    for (int j = 0; j < batch; j++) {
                        int min_ch = -1;
                        float min_v = float.MaxValue;

                        for (int k = 0; k < channels; k++) {
                            float v = x1[k + j * channels];

                            if (min_v > v) {
                                min_v = v;
                                min_ch = k;
                            }
                        }

                        Assert.AreEqual(min_ch, y[j], 1e-6f, $"batch:{batch},channels:{channels} idx:{j}");
                    }
                }
            }
        }
    }
}
