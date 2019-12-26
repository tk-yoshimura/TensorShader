using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class SpaceToChannelTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 5 }) {
                    foreach (int scale in new int[] { 2, 3, 4 }) {
                        foreach (int outwidth in new int[] { 5, 7, 11 }) {
                            foreach (int outheight in new int[] { 5, 7, 11 }) {
                                foreach (int outdepth in new int[] { 5, 7, 11 }) {
                                    int inwidth = outwidth * scale, inheight = outheight * scale, indepth = outdepth * scale, outchannels = inchannels * scale * scale * scale;

                                    float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                    Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, batch, xval);

                                    Map3D y = Reference(x, scale);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                                    SpaceToChannel ope = new SpaceToChannel(inwidth, inheight, indepth, inchannels, scale, batch);

                                    ope.Execute(x_tensor, y_tensor);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{scale},{inwidth},{inheight},{indepth},{batch}");

                                    Console.WriteLine($"pass: {inchannels},{scale},{inwidth},{inheight},{indepth},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 32, scale = 2;
            int outwidth = inwidth / scale, outheight = inheight / scale, outdepth = indepth / scale, outchannels = inchannels * (scale * scale * scale);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            SpaceToChannel ope = new SpaceToChannel(inwidth, inheight, indepth, inchannels, scale);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/space_to_channel_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D x, int scale) {
            int inw = x.Width, inh = x.Height, ind = x.Depth, inchannels = x.Channels, batch = x.Batch;
            if (inw % scale != 0 || inh % scale != 0 || ind % scale != 0) {
                throw new ArgumentException(nameof(scale));
            }

            int outw = inw / scale, outh = inh / scale, outd = ind / scale;
            int outchannels = inchannels * scale * scale * scale;

            Map3D y = new Map3D(outchannels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy, oz = 0; oz < outd; oz++) {
                    for (oy = 0; oy < outh; oy++) {
                        for (ox = 0; ox < outw; ox++) {
                            for (int kx, ky, kz = 0; kz < scale; kz++) {
                                for (ky = 0; ky < scale; ky++) {
                                    for (kx = 0; kx < scale; kx++) {
                                        for (int inch = 0; inch < inchannels; inch++) {
                                            int outch = inch + (kx + ky * scale + kz * scale * scale) * inchannels;

                                            y[outch, ox, oy, oz, th] = x[inch, ox * scale + kx, oy * scale + ky, oz * scale + kz, th];

                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 32, scale = 2, inwidth = 7, inheight = 5, indepth = 3;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();

            Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, 1, xval);

            Map3D y = Reference(ChannelToSpaceTest.Reference(x, scale), scale);

            CollectionAssert.AreEqual(x.ToArray(), y.ToArray());
        }
    }
}
