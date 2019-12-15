using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.QuaternionCast;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionCastTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1024 - 4; length <= i * 1024 + 4; length += 1) {
                    if (length < 1) continue;

                    float[] x = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] y = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] z = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] w = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape inshape = Shape.Vector(length);
                    Shape outshape = Shape.Vector(length * 4);

                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(inshape, x);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(inshape, y);
                    OverflowCheckedTensor v3 = new OverflowCheckedTensor(inshape, z);
                    OverflowCheckedTensor v4 = new OverflowCheckedTensor(inshape, w);
                    OverflowCheckedTensor v5 = new OverflowCheckedTensor(outshape);

                    QuaternionCast ope = new QuaternionCast(inshape);

                    ope.Execute(v1, v2, v3, v4, v5);

                    CollectionAssert.AreEqual(x, v1.State);
                    CollectionAssert.AreEqual(y, v2.State);
                    CollectionAssert.AreEqual(z, v3.State);
                    CollectionAssert.AreEqual(w, v4.State);

                    float[] v = v5.State;

                    for (int j = 0; j < length; j++) {
                        Assert.AreEqual(x[j], v[j * 4], 1e-6f, $"length:{length}, idx:{j}");
                        Assert.AreEqual(y[j], v[j * 4 + 1], 1e-6f, $"length:{length}, idx:{j}");
                        Assert.AreEqual(z[j], v[j * 4 + 2], 1e-6f, $"length:{length}, idx:{j}");
                        Assert.AreEqual(w[j], v[j * 4 + 3], 1e-6f, $"length:{length}, idx:{j}");
                    }
                };
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 65536 * 4;

            Shape inshape = Shape.Vector(length);
            Shape outshape = Shape.Vector(length * 4);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(inshape);
            OverflowCheckedTensor v2 = new OverflowCheckedTensor(inshape);
            OverflowCheckedTensor v3 = new OverflowCheckedTensor(inshape);
            OverflowCheckedTensor v4 = new OverflowCheckedTensor(inshape);
            OverflowCheckedTensor v5 = new OverflowCheckedTensor(outshape);

            QuaternionCast ope = new QuaternionCast(inshape);

            Stopwatch sw = new Stopwatch();

            sw.Start();

            ope.Execute(v1, v2, v3, v4, v5);
            ope.Execute(v1, v2, v3, v4, v5);
            ope.Execute(v1, v2, v3, v4, v5);
            ope.Execute(v1, v2, v3, v4, v5);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }
    }
}
