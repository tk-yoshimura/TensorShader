using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.TrivectorCast;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorCastTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new(1234);

            for (int i = 0; i < 64; i++) {
                for (int length = i * 1024 - 4; length <= i * 1024 + 4; length += 1) {
                    if (length < 1) continue;

                    float[] x = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] y = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] z = (new float[length]).Select((_) => (float)rd.NextDouble()).ToArray();

                    Shape inshape = Shape.Vector(length);
                    Shape outshape = Shape.Vector(length * 3);

                    OverflowCheckedTensor v1 = new(inshape, x);
                    OverflowCheckedTensor v2 = new(inshape, y);
                    OverflowCheckedTensor v3 = new(inshape, z);
                    OverflowCheckedTensor v4 = new(outshape);

                    TrivectorCast ope = new(inshape);

                    ope.Execute(v1, v2, v3, v4);

                    CollectionAssert.AreEqual(x, v1.State.Value);
                    CollectionAssert.AreEqual(y, v2.State.Value);
                    CollectionAssert.AreEqual(z, v3.State.Value);

                    float[] v = v4.State.Value;

                    for (int j = 0; j < length; j++) {
                        Assert.AreEqual(x[j], v[j * 3], 1e-6f, $"length:{length}, idx:{j}");
                        Assert.AreEqual(y[j], v[j * 3 + 1], 1e-6f, $"length:{length}, idx:{j}");
                        Assert.AreEqual(z[j], v[j * 3 + 2], 1e-6f, $"length:{length}, idx:{j}");
                    }
                };
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 65536 * 4;

            Shape inshape = Shape.Vector(length);
            Shape outshape = Shape.Vector(length * 3);

            OverflowCheckedTensor v1 = new(inshape);
            OverflowCheckedTensor v2 = new(inshape);
            OverflowCheckedTensor v3 = new(inshape);
            OverflowCheckedTensor v4 = new(outshape);

            TrivectorCast ope = new(inshape);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trivector_cast.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(v1, v2, v3, v4);

            Cuda.Profiler.Stop();
        }
    }
}
