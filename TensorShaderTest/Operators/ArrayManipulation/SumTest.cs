using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class SumTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (int n = 1; n <= 8; n++) {
                for (int i = 0; i < 32; i++) {
                    for (int length = i * 1024 - 2; length <= i * 1024 + 2; length++) {
                        if (length < 1) continue;

                        Shape shape = Shape.Vector(length);
                        float[][] xs = new float[n][];
                        OverflowCheckedTensor[] vs = new OverflowCheckedTensor[n];
                        OverflowCheckedTensor u = new OverflowCheckedTensor(shape);

                        for (int j = 0; j < n; j++) {
                            xs[j] = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                            vs[j] = new OverflowCheckedTensor(shape, xs[j]);
                        }

                       

                        Sum ope = new Sum(shape, n);

                        Cuda.Profiler.Start();

                        ope.Execute(vs.Concat(new Tensor[] { u }).ToArray());

                        Cuda.Profiler.Stop();

                        for (int j = 0; j < n; j++) {
                            CollectionAssert.AreEqual(xs[j], vs[j].State);
                        }

                        float[] y = u.State;

                        for (int j = 0; j < length; j++) {
                            float sum = 0;
                            for (int k = 0; k < n; k++) {
                                sum += xs[k][j];
                            }

                            Assert.AreEqual(sum, y[j], 1e-5f, $"length:{length}, idx:{j}");
                        }
                    };
                }
            }
        }
    }
}
