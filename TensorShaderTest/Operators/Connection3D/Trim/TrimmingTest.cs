using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class TrimmingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 6, 7, 8 }) {
                    foreach ((int lefttrim, int righttrim) in new (int, int)[] { (0, 0), (1, 2), (2, 1) }) {
                        foreach ((int toptrim, int bottomtrim) in new (int, int)[] { (0, 0), (1, 2), (2, 1) }) {
                            foreach ((int fronttrim, int reartrim) in new (int, int)[] { (0, 0), (1, 2), (2, 1) }) {
                                foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (6, 6, 9), (6, 9, 6), (9, 6, 6) }) {
                                    int outwidth = inwidth - lefttrim - righttrim, outheight = inheight - toptrim - bottomtrim, outdepth = indepth - fronttrim - reartrim;

                                    float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                    Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);

                                    Map3D y = Reference(x, lefttrim, righttrim, toptrim, bottomtrim, fronttrim, reartrim);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

                                    Trimming ope = new Trimming(inwidth, inheight, indepth, channels, lefttrim, righttrim, toptrim, bottomtrim, fronttrim, reartrim, batch);

                                    ope.Execute(x_tensor, y_tensor);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_tensor.State.Value;

                                    CollectionAssert.AreEqual(xval, x_tensor.State.Value);

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{lefttrim},{righttrim},{toptrim},{bottomtrim},{fronttrim},{reartrim},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {channels},{lefttrim},{righttrim},{toptrim},{bottomtrim},{fronttrim},{reartrim},{inwidth},{inheight},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D x, int lefttrim, int righttrim, int toptrim, int bottomtrim, int fronttrim, int reartrim) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - lefttrim - righttrim, outh = inh - toptrim - bottomtrim, outd = ind - fronttrim - reartrim;

            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy, oz = 0; oz < outd; oz++) {
                    for (oy = 0; oy < outh; oy++) {
                        for (ox = 0; ox < outw; ox++) {
                            for (int ch = 0; ch < channels; ch++) {
                                y[ch, ox, oy, oz, th] = x[ch, ox + lefttrim, oy + toptrim, oz + fronttrim, th];
                            }
                        }
                    }
                }

            }

            return y;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, channels = 32, lefttrim = 1, righttrim = 1, toptrim = 1, bottomtrim = 1, fronttrim = 1, reartrim = 1, batch = 4;
            int outwidth = inwidth - lefttrim - righttrim, outheight = inheight - toptrim - bottomtrim, outdepth = indepth - fronttrim - reartrim;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

            Trimming ope = new Trimming(inwidth, inheight, indepth, channels, lefttrim, righttrim, toptrim, bottomtrim, fronttrim, reartrim, batch);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/trimming_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}
