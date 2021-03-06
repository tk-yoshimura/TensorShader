using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class ImageToColumnTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                            Map3D x = new(channels, inwidth, inheight, indepth, batch, xval);

                            Map3D y = Reference(x, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);

                            OverflowCheckedTensor y_tensor = new(new Shape(ShapeType.Column, kwidth * kheight * kdepth, channels, outwidth, outheight, outdepth, batch));

                            ImageToColumn ope = new(inwidth, inheight, indepth, channels, kwidth, kheight, kdepth, batch);

                            ope.Execute(x_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 32, inheight = 32, indepth = 32, channels = 31, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth));

            OverflowCheckedTensor y_tensor = new(new Shape(ShapeType.Column, ksize * ksize * ksize, channels, outwidth, outheight, outdepth, 1));

            ImageToColumn ope = new(inwidth, inheight, indepth, channels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/image_to_column_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D x, int kwidth, int kheight, int kdepth) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            Map3D y = new(kwidth * kheight * kdepth * channels, outw, outh, outd, batch);

            for (int k = 0, kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; k++, kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        for (int ch = 0; ch < channels; ch++) {
                                            y[k + kwidth * kheight * kdepth * ch, ox, oy, oz, th] = x[ch, kx + ox, ky + oy, kz + oz, th];
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
    }
}
