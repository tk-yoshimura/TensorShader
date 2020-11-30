using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Indexer;

namespace TensorShaderTest.Operators.Indexer {
    [TestClass]
    public class OneHotVectorTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            foreach (int batch in new int[] { 1, 2, 3, 4, 8, 16, 32 }) {
                for (int channels = 1; channels <= 1024; channels *= 2) {
                    float[] x1 = (new float[batch]).Select((_) => (float)rd.Next(channels)).ToArray();

                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(Shape.Vector(batch), x1);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(Shape.Map0D(channels, batch));

                    OneHotVector ope = new OneHotVector(channels, v1.Shape);

                    ope.Execute(v1, v2);

                    CollectionAssert.AreEqual(x1, v1.State.Value);

                    float[] y = v2.State.Value;

                    for (int j = 0; j < batch; j++) {
                        for (int k = 0; k < channels; k++) {
                            Assert.AreEqual(k == x1[j] ? 1 : 0, y[k + j * channels], $"batch:{batch},channels:{channels} idx:{j}");
                        }
                    }

                    Assert.AreEqual(batch, y.Sum());
                }
            }
        }
    }
}
