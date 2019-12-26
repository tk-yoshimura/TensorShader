using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class LinearZoomTest {
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

                                LinearZoom ope = new LinearZoom(inwidth, inheight, indepth, channels, batch);

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

            LinearZoom ope = new LinearZoom(inwidth, inheight, indepth, channels);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/linearzoom_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map3D Reference(Map3D x, int scale) {
            int inw = x.Width, inh = x.Height, ind = x.Depth, channels = x.Channels, batch = x.Batch;

            int outw = inw * scale, outh = inh * scale, outd = ind * scale;
            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy, iz = 0; iz < ind; iz++) {
                    for (iy = 0; iy < inh; iy++) {
                        for (ix = 0; ix < inw; ix++) {
                            int x0 = ix, xm = Math.Max(0, ix - 1), xp = Math.Min(inw - 1, ix + 1);
                            int y0 = iy, ym = Math.Max(0, iy - 1), yp = Math.Min(inh - 1, iy + 1);
                            int z0 = iz, zm = Math.Max(0, iz - 1), zp = Math.Min(ind - 1, iz + 1);

                            for (int f = 0; f < channels; f++) {
                                double vc = x[f, x0, y0, z0, th] * 8;

                                double vxm = x[f, xm, y0, z0, th] * 4;
                                double vxp = x[f, xp, y0, z0, th] * 4;
                                double vym = x[f, x0, ym, z0, th] * 4;
                                double vyp = x[f, x0, yp, z0, th] * 4;
                                double vzm = x[f, x0, y0, zm, th] * 4;
                                double vzp = x[f, x0, y0, zp, th] * 4;

                                double vxmym = x[f, xm, ym, z0, th] * 2;
                                double vxpym = x[f, xp, ym, z0, th] * 2;
                                double vxmyp = x[f, xm, yp, z0, th] * 2;
                                double vxpyp = x[f, xp, yp, z0, th] * 2;
                                double vymzm = x[f, x0, ym, zm, th] * 2;
                                double vypzm = x[f, x0, yp, zm, th] * 2;
                                double vymzp = x[f, x0, ym, zp, th] * 2;
                                double vypzp = x[f, x0, yp, zp, th] * 2;
                                double vxmzm = x[f, xm, y0, zm, th] * 2;
                                double vxpzm = x[f, xp, y0, zm, th] * 2;
                                double vxmzp = x[f, xm, y0, zp, th] * 2;
                                double vxpzp = x[f, xp, y0, zp, th] * 2;

                                double vxmymzm = x[f, xm, ym, zm, th];
                                double vxpymzm = x[f, xp, ym, zm, th];
                                double vxmypzm = x[f, xm, yp, zm, th];
                                double vxpypzm = x[f, xp, yp, zm, th];
                                double vxmymzp = x[f, xm, ym, zp, th];
                                double vxpymzp = x[f, xp, ym, zp, th];
                                double vxmypzp = x[f, xm, yp, zp, th];
                                double vxpypzp = x[f, xp, yp, zp, th];

                                y[f, ix * 2, iy * 2, iz * 2, th] = (vc + vxm + vym + vzm + vxmym + vymzm + vxmzm + vxmymzm) / 27;
                                y[f, ix * 2 + 1, iy * 2, iz * 2, th] = (vc + vxp + vym + vzm + vxpym + vymzm + vxpzm + vxpymzm) / 27;
                                y[f, ix * 2, iy * 2 + 1, iz * 2, th] = (vc + vxm + vyp + vzm + vxmyp + vypzm + vxmzm + vxmypzm) / 27;
                                y[f, ix * 2 + 1, iy * 2 + 1, iz * 2, th] = (vc + vxp + vyp + vzm + vxpyp + vypzm + vxpzm + vxpypzm) / 27;
                                y[f, ix * 2, iy * 2, iz * 2 + 1, th] = (vc + vxm + vym + vzp + vxmym + vymzp + vxmzp + vxmymzp) / 27;
                                y[f, ix * 2 + 1, iy * 2, iz * 2 + 1, th] = (vc + vxp + vym + vzp + vxpym + vymzp + vxpzp + vxpymzp) / 27;
                                y[f, ix * 2, iy * 2 + 1, iz * 2 + 1, th] = (vc + vxm + vyp + vzp + vxmyp + vypzp + vxmzp + vxmypzp) / 27;
                                y[f, ix * 2 + 1, iy * 2 + 1, iz * 2 + 1, th] = (vc + vxp + vyp + vzp + vxpyp + vypzp + vxpzp + vxpypzp) / 27;

                            }
                        }
                    }
                }
            }

            return y;
        }
    }
}
