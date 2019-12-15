using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class NeighborZoomTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            int scale = 2;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }) {
                    foreach (int inwidth in new int[] { 1, 2, 5, 7, 11 }) {
                        foreach (int inheight in new int[] { 1, 2, 5, 7, 11 }) {
                            foreach (int indepth in new int[] { 1, 2, 5, 7, 11 }) {
                                int outwidth = inwidth * scale, outheight = inheight * scale, outdepth = indepth * scale;

                                float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);

                                Map3D y = Reference(x, scale);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

                                NeighborZoom ope = new NeighborZoom(inwidth, inheight, indepth, channels, batch);

                                ope.Execute(x_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{inwidth},{inheight},{indepth},{batch}");

                                Console.WriteLine($"pass: {channels},{inwidth},{inheight},{indepth},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, channels = 32, scale = 2;
            int outwidth = inwidth * scale, outheight = inheight * scale, outdepth = indepth * scale;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth));

            NeighborZoom ope = new NeighborZoom(inwidth, inheight, indepth, channels);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/neighborzoom3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D x, int scale) {
            int inw = x.Width, inh = x.Height, ind = x.Depth, channels = x.Channels, batch = x.Batch;

            int outw = inw * scale, outh = inh * scale, outd = ind * scale;
            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy, oz = 0; oz < outd; oz++) {
                    for (oy = 0; oy < outh; oy++) {
                        for (ox = 0; ox < outw; ox++) {
                            for (int f = 0; f < channels; f++) {
                                y[f, ox, oy, oz, th] = x[f, ox / 2, oy / 2, oz / 2, th];
                            }

                        }
                    }
                }

            }

            return y;
        }
    }
}
