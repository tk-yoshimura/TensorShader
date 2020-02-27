using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class NeighborZoomTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            int scale = 2;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }) {
                    foreach (int inwidth in new int[] { 1, 2, 5, 7, 11 }) {
                        int outwidth = inwidth * scale;

                        float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                        Map1D x = new Map1D(channels, inwidth, batch, xval);

                        Map1D y = Reference(x, scale);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth, batch), xval);
                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth, batch));

                        NeighborZoom ope = new NeighborZoom(inwidth, channels, batch);

                        ope.Execute(x_tensor, y_tensor);

                        float[] y_expect = y.ToArray();
                        float[] y_actual = y_tensor.State;

                        CollectionAssert.AreEqual(xval, x_tensor.State);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{inwidth},{batch}");

                        Console.WriteLine($"pass: {channels},{inwidth},{batch}");

                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, channels = 32, scale = 2;
            int outwidth = inwidth * scale;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, inwidth));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(channels, outwidth));

            NeighborZoom ope = new NeighborZoom(inwidth, channels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/neighborzoom_1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D x, int scale) {
            int inw = x.Width, channels = x.Channels, batch = x.Batch;

            int outw = inw * scale;
            Map1D y = new Map1D(channels, outw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox = 0; ox < outw; ox++) {
                    for (int f = 0; f < channels; f++) {
                        y[f, ox, th] = x[f, ox / 2, th];
                    }
                }

            }

            return y;
        }
    }
}
