using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.ComplexCast;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexCastTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1024 - 4; length <= i * 1024 + 4; length += 1) {
                    if (length < 1) continue;

                    float[] x = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] y = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape inshape = Shape.Vector(length);
                    Shape outshape = Shape.Vector(length * 2);

                    OverflowCheckedTensor v1 = new(inshape, x);
                    OverflowCheckedTensor v2 = new(inshape, y);
                    OverflowCheckedTensor v3 = new(outshape);

                    ComplexCast ope = new(inshape);

                    ope.Execute(v1, v2, v3);

                    CollectionAssert.AreEqual(x, v1.State.Value);
                    CollectionAssert.AreEqual(y, v2.State.Value);

                    float[] v = v3.State.Value;

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

            OverflowCheckedTensor v1 = new(inshape);
            OverflowCheckedTensor v2 = new(inshape);
            OverflowCheckedTensor v3 = new(outshape);

            ComplexCast ope = new(inshape);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/complex_cast.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(v1, v2, v3);

            Cuda.Profiler.Stop();
        }
    }
}
