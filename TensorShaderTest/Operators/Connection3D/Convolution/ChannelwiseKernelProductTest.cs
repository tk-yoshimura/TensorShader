using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class ChannelwiseKernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] gyval = (new float[outwidth * outheight * outdepth * channels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new(channels, inwidth, inheight, indepth, batch, xval);
                            Map3D gy = new(channels, outwidth, outheight, outdepth, batch, gyval);

                            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor gy_tensor = new(Shape.Map3D(channels, outwidth, outheight, outdepth, batch), gyval);

                            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(channels, 1, kwidth, kheight, kdepth));

                            ChannelwiseKernelProduct ope = new(inwidth, inheight, indepth, channels, kwidth, kheight, kdepth, batch);

                            ope.Execute(x_tensor, gy_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            float max_err = 0;

            Random random = new(1234);

            int batch = 3;
            int channels = 49;
            int kwidth = 7, kheight = 5, kdepth = 3;
            int inwidth = 125, inheight = 196, indepth = 4;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outdepth * channels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map3D x = new(channels, inwidth, inheight, indepth, batch, xval);
            Map3D gy = new(channels, outwidth, outheight, outdepth, batch, gyval);

            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
            OverflowCheckedTensor gy_tensor = new(Shape.Map3D(channels, outwidth, outheight, outdepth, batch), gyval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(channels, 1, kwidth, kheight, kdepth));

            ChannelwiseKernelProduct ope = new(inwidth, inheight, indepth, channels, kwidth, kheight, kdepth, batch);

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, channels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(channels, inwidth, inheight, indepth));
            OverflowCheckedTensor gy_tensor = new(Shape.Map3D(channels, outwidth, outheight, outdepth));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel3D(channels, 1, ksize, ksize, ksize));

            ChannelwiseKernelProduct ope = new(inwidth, inheight, indepth, channels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/chwise_kernelproduct_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter3D Reference(Map3D x, Map3D gy, int kwidth, int kheight, int kdepth) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1 || outd != ind - kdepth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter3D w = new(channels, 1, kwidth, kheight, kdepth);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ix, iy, iz = kz, ox, oy, oz = 0; oz < outd; iz++, oz++) {
                                for (iy = ky, oy = 0; oy < outh; iy++, oy++) {
                                    for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                        for (int ch = 0; ch < channels; ch++) {
                                            w[ch, 0, kx, ky, kz] += x[ch, ix, iy, iz, th] * gy[ch, ox, oy, oz, th];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 2, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 13, inheight = 12, indepth = 11;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * channels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outdepth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new(channels, inwidth, inheight, indepth, 1, xval);
            Map3D gy = new(channels, outwidth, outheight, outdepth, 1, gyval);

            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth);

            float[] gw_expect = {
                9.06985200e+01f,  9.05726800e+01f,  9.10857200e+01f,  9.09590000e+01f,  9.14729200e+01f,  9.13453200e+01f,  9.57321200e+01f,
                9.55948400e+01f,  9.61193200e+01f,  9.59811600e+01f,  9.65065200e+01f,  9.63674800e+01f,  1.00765720e+02f,  1.00617000e+02f,
                1.01152920e+02f,  1.01003320e+02f,  1.01540120e+02f,  1.01389640e+02f,  1.05799320e+02f,  1.05639160e+02f,  1.06186520e+02f,
                1.06025480e+02f,  1.06573720e+02f,  1.06411800e+02f,  1.10832920e+02f,  1.10661320e+02f,  1.11220120e+02f,  1.11047640e+02f,
                1.11607320e+02f,  1.11433960e+02f,  1.51101720e+02f,  1.50838600e+02f,  1.51488920e+02f,  1.51224920e+02f,  1.51876120e+02f,
                1.51611240e+02f,  1.56135320e+02f,  1.55860760e+02f,  1.56522520e+02f,  1.56247080e+02f,  1.56909720e+02f,  1.56633400e+02f,
                1.61168920e+02f,  1.60882920e+02f,  1.61556120e+02f,  1.61269240e+02f,  1.61943320e+02f,  1.61655560e+02f,  1.66202520e+02f,
                1.65905080e+02f,  1.66589720e+02f,  1.66291400e+02f,  1.66976920e+02f,  1.66677720e+02f,  1.71236120e+02f,  1.70927240e+02f,
                1.71623320e+02f,  1.71313560e+02f,  1.72010520e+02f,  1.71699880e+02f,  2.11504920e+02f,  2.11104520e+02f,  2.11892120e+02f,
                2.11490840e+02f,  2.12279320e+02f,  2.11877160e+02f,  2.16538520e+02f,  2.16126680e+02f,  2.16925720e+02f,  2.16513000e+02f,
                2.17312920e+02f,  2.16899320e+02f,  2.21572120e+02f,  2.21148840e+02f,  2.21959320e+02f,  2.21535160e+02f,  2.22346520e+02f,
                2.21921480e+02f,  2.26605720e+02f,  2.26171000e+02f,  2.26992920e+02f,  2.26557320e+02f,  2.27380120e+02f,  2.26943640e+02f,
                2.31639320e+02f,  2.31193160e+02f,  2.32026520e+02f,  2.31579480e+02f,  2.32413720e+02f,  2.31965800e+02f,  2.71908120e+02f,
                2.71370440e+02f,  2.72295320e+02f,  2.71756760e+02f,  2.72682520e+02f,  2.72143080e+02f,  2.76941720e+02f,  2.76392600e+02f,
                2.77328920e+02f,  2.76778920e+02f,  2.77716120e+02f,  2.77165240e+02f,  2.81975320e+02f,  2.81414760e+02f,  2.82362520e+02f,
                2.81801080e+02f,  2.82749720e+02f,  2.82187400e+02f,  2.87008920e+02f,  2.86436920e+02f,  2.87396120e+02f,  2.86823240e+02f,
                2.87783320e+02f,  2.87209560e+02f,  2.92042520e+02f,  2.91459080e+02f,  2.92429720e+02f,  2.91845400e+02f,  2.92816920e+02f,
                2.92231720e+02f,  3.32311320e+02f,  3.31636360e+02f,  3.32698520e+02f,  3.32022680e+02f,  3.33085720e+02f,  3.32409000e+02f,
                3.37344920e+02f,  3.36658520e+02f,  3.37732120e+02f,  3.37044840e+02f,  3.38119320e+02f,  3.37431160e+02f,  3.42378520e+02f,
                3.41680680e+02f,  3.42765720e+02f,  3.42067000e+02f,  3.43152920e+02f,  3.42453320e+02f,  3.47412120e+02f,  3.46702840e+02f,
                3.47799320e+02f,  3.47089160e+02f,  3.48186520e+02f,  3.47475480e+02f,  3.52445720e+02f,  3.51725000e+02f,  3.52832920e+02f,
                3.52111320e+02f,  3.53220120e+02f,  3.52497640e+02f,  3.92714520e+02f,  3.91902280e+02f,  3.93101720e+02f,  3.92288600e+02f,
                3.93488920e+02f,  3.92674920e+02f,  3.97748120e+02f,  3.96924440e+02f,  3.98135320e+02f,  3.97310760e+02f,  3.98522520e+02f,
                3.97697080e+02f,  4.02781720e+02f,  4.01946600e+02f,  4.03168920e+02f,  4.02332920e+02f,  4.03556120e+02f,  4.02719240e+02f,
                4.07815320e+02f,  4.06968760e+02f,  4.08202520e+02f,  4.07355080e+02f,  4.08589720e+02f,  4.07741400e+02f,  4.12848920e+02f,
                4.11990920e+02f,  4.13236120e+02f,  4.12377240e+02f,  4.13623320e+02f,  4.12763560e+02f,  4.53117720e+02f,  4.52168200e+02f,
                4.53504920e+02f,  4.52554520e+02f,  4.53892120e+02f,  4.52940840e+02f,  4.58151320e+02f,  4.57190360e+02f,  4.58538520e+02f,
                4.57576680e+02f,  4.58925720e+02f,  4.57963000e+02f,  4.63184920e+02f,  4.62212520e+02f,  4.63572120e+02f,  4.62598840e+02f,
                4.63959320e+02f,  4.62985160e+02f,  4.68218520e+02f,  4.67234680e+02f,  4.68605720e+02f,  4.67621000e+02f,  4.68992920e+02f,
                4.68007320e+02f,  4.73252120e+02f,  4.72256840e+02f,  4.73639320e+02f,  4.72643160e+02f,  4.74026520e+02f,  4.73029480e+02f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {channels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth}");
        }
    }
}
