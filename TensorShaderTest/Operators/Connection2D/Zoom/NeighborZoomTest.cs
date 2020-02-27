using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class NeighborZoomTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            int scale = 2;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }) {
                    foreach (int inheight in new int[] { 1, 2, 5, 7, 11 }) {
                        foreach (int inwidth in new int[] { 1, 2, 5, 7, 11 }) {
                            int outwidth = inwidth * scale, outheight = inheight * scale;

                            float[] xval = (new float[inwidth * inheight * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                            Map2D x = new Map2D(channels, inwidth, inheight, batch, xval);

                            Map2D y = Reference(x, scale);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight, batch), xval);
                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight, batch));

                            NeighborZoom ope = new NeighborZoom(inwidth, inheight, channels, batch);

                            ope.Execute(x_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State;

                            CollectionAssert.AreEqual(xval, x_tensor.State);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{inwidth},{inheight},{batch}");

                            Console.WriteLine($"pass: {channels},{inwidth},{inheight},{batch}");

                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, channels = 32, scale = 2;
            int outwidth = inwidth * scale, outheight = inheight * scale;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, inwidth, inheight));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(channels, outwidth, outheight));

            NeighborZoom ope = new NeighborZoom(inwidth, inheight, channels);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/neighborzoom_2d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map2D Reference(Map2D x, int scale) {
            int inw = x.Width, inh = x.Height, channels = x.Channels, batch = x.Batch;

            int outw = inw * scale, outh = inh * scale;
            Map2D y = new Map2D(channels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy = 0; oy < outh; oy++) {
                    for (ox = 0; ox < outw; ox++) {
                        for (int f = 0; f < channels; f++) {
                            y[f, ox, oy, th] = x[f, ox / 2, oy / 2, th];
                        }

                    }
                }

            }

            return y;
        }
    }
}
