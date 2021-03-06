using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Indexer;

namespace TensorShaderTest.Operators.Indexer {
    [TestClass]
    public class ArgMaxTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new(1234);

            foreach (int batch in new int[] { 1, 2, 3, 4, 8, 16, 32 }) {
                for (int channels = 1; channels <= 1024; channels *= 2) {
                    float[] x1 = (new float[channels * batch]).Select((_) => (float)rd.NextDouble()).ToArray();

                    OverflowCheckedTensor v1 = new(Shape.Map0D(channels, batch), x1);
                    OverflowCheckedTensor v2 = new(Shape.Vector(batch));

                    ArgMax ope = new(channels, batch);

                    ope.Execute(v1, v2);

                    CollectionAssert.AreEqual(x1, v1.State.Value);

                    float[] y = v2.State.Value;

                    for (int j = 0; j < batch; j++) {
                        int max_ch = -1;
                        float max_v = float.MinValue;

                        for (int k = 0; k < channels; k++) {
                            float v = x1[k + j * channels];

                            if (max_v < v) {
                                max_v = v;
                                max_ch = k;
                            }
                        }

                        Assert.AreEqual(max_ch, y[j], 1e-6f, $"batch:{batch},channels:{channels} idx:{j}");
                    }
                }
            }
        }
    }
}
