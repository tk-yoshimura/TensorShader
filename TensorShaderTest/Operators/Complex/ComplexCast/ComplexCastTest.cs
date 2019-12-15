using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexCast;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexCastTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1024 - 4; length <= i * 1024 + 4; length += 1) {
                    if (length < 1) continue;

                    float[] x = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] y = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape inshape = Shape.Vector(length);
                    Shape outshape = Shape.Vector(length * 2);

                    OverflowCheckedTensor v1 = new OverflowCheckedTensor(inshape, x);
                    OverflowCheckedTensor v2 = new OverflowCheckedTensor(inshape, y);
                    OverflowCheckedTensor v3 = new OverflowCheckedTensor(outshape);

                    ComplexCast ope = new ComplexCast(inshape);

                    ope.Execute(v1, v2, v3);

                    CollectionAssert.AreEqual(x, v1.State);
                    CollectionAssert.AreEqual(y, v2.State);

                    float[] v = v3.State;

                    for (int j = 0; j < length; j++) {
                        Assert.AreEqual(x[j], v[j * 2], 1e-6f, $"length:{length}, idx:{j}");
                        Assert.AreEqual(y[j], v[j * 2 + 1], 1e-6f, $"length:{length}, idx:{j}");
                    }
                };
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 65536 * 4;

            Shape inshape = Shape.Vector(length);
            Shape outshape = Shape.Vector(length * 2);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(inshape);
            OverflowCheckedTensor v2 = new OverflowCheckedTensor(inshape);
            OverflowCheckedTensor v3 = new OverflowCheckedTensor(outshape);

            ComplexCast ope = new ComplexCast(inshape);

            Stopwatch sw = new Stopwatch();

            sw.Start();

            ope.Execute(v1, v2, v3);
            ope.Execute(v1, v2, v3);
            ope.Execute(v1, v2, v3);
            ope.Execute(v1, v2, v3);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }
    }
}
