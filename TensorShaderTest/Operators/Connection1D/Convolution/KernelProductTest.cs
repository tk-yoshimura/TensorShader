using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class KernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int stride in new int[] { 1 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    int outwidth = inwidth - kwidth + 1;

                                    float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] gyval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map1D x = new Map1D(inchannels, inwidth, batch, xval);
                                    Map1D gy = new Map1D(outchannels, outwidth, batch, gyval);

                                    Filter1D gw = Reference(x, gy, kwidth, stride);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                    OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), gyval);

                                    OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels, kwidth));

                                    KernelProduct ope = new KernelProduct(inwidth, inchannels, outchannels, kwidth, stride, batch);

                                    ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(gyval, gy_tensor.State);

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

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
            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3, stride = 1;
            int outwidth = (inwidth - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels, ksize));

            KernelProduct ope = new KernelProduct(inwidth, inchannels, outchannels, ksize, stride);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_1d_atomic_xset16_3.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            
            Cuda.Profiler.Stop();
        }

        public static Filter1D Reference(Map1D x, Map1D gy, int kwidth, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != inw - kwidth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter1D w = new Filter1D(inchannels, outchannels, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int inch, outch = 0; outch < outchannels; outch++) {
                        for (inch = 0; inch < inchannels; inch++) {
                            double sum = 0;

                            for (int ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                                sum += x[inch, ix, th] * gy[outch, ox, th];
                            }

                            w[inch, outch, kx] += sum;
                        }
                    }
                }
            }

            return w;
        }

        public static Filter1D OptimizedReference(Map1D x, Map1D gy, int kwidth, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw < inw - kwidth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter1D w = new Filter1D(outchannels, inchannels, kwidth);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int outch, inch = 0; inch < inchannels; inch++) {
                        for (outch = 0; outch < outchannels; outch++) {
                            int filter_idx = inch + inchannels * outch + kx * inchannels * outchannels;
                            int inmap_idx = inch + kx * inchannels + th * inw * inchannels;
                            int outmap_idx = outch + th * outw * outchannels;

                            double sum = 0;

                            for (int ox = 0; ox < outw; ox++) {
                                sum += x[inmap_idx] * gy[outmap_idx];

                                inmap_idx += inchannels * stride;
                                outmap_idx += outchannels;
                            }

                            w[filter_idx] += sum;
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, stride = 2, inwidth = 13;
            int outwidth = inwidth - kwidth + 1;

            float[] xval = (new float[inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new Map1D(inchannels, inwidth, 1, xval);
            Map1D gy = new Map1D(outchannels, outwidth, 1, gyval);

            Filter1D gw = Reference(x, gy, kwidth, stride);

            float[] gw_expect = {
                5.180000e-03f, 5.405000e-03f, 5.630000e-03f, 5.855000e-03f, 6.080000e-03f,
                6.305000e-03f, 6.530000e-03f, 4.970000e-03f, 5.189000e-03f, 5.408000e-03f,
                5.627000e-03f, 5.846000e-03f, 6.065000e-03f, 6.284000e-03f, 4.760000e-03f,
                4.973000e-03f, 5.186000e-03f, 5.399000e-03f, 5.612000e-03f, 5.825000e-03f,
                6.038000e-03f, 4.550000e-03f, 4.757000e-03f, 4.964000e-03f, 5.171000e-03f,
                5.378000e-03f, 5.585000e-03f, 5.792000e-03f, 4.340000e-03f, 4.541000e-03f,
                4.742000e-03f, 4.943000e-03f, 5.144000e-03f, 5.345000e-03f, 5.546000e-03f,
                4.130000e-03f, 4.325000e-03f, 4.520000e-03f, 4.715000e-03f, 4.910000e-03f,
                5.105000e-03f, 5.300000e-03f, 3.920000e-03f, 4.109000e-03f, 4.298000e-03f,
                4.487000e-03f, 4.676000e-03f, 4.865000e-03f, 5.054000e-03f, 3.710000e-03f,
                3.893000e-03f, 4.076000e-03f, 4.259000e-03f, 4.442000e-03f, 4.625000e-03f,
                4.808000e-03f, 3.500000e-03f, 3.677000e-03f, 3.854000e-03f, 4.031000e-03f,
                4.208000e-03f, 4.385000e-03f, 4.562000e-03f, 3.290000e-03f, 3.461000e-03f,
                3.632000e-03f, 3.803000e-03f, 3.974000e-03f, 4.145000e-03f, 4.316000e-03f,
                3.080000e-03f, 3.245000e-03f, 3.410000e-03f, 3.575000e-03f, 3.740000e-03f,
                3.905000e-03f, 4.070000e-03f, 6.755000e-03f, 6.980000e-03f, 7.205000e-03f,
                7.430000e-03f, 7.655000e-03f, 7.880000e-03f, 8.105000e-03f, 6.503000e-03f,
                6.722000e-03f, 6.941000e-03f, 7.160000e-03f, 7.379000e-03f, 7.598000e-03f,
                7.817000e-03f, 6.251000e-03f, 6.464000e-03f, 6.677000e-03f, 6.890000e-03f,
                7.103000e-03f, 7.316000e-03f, 7.529000e-03f, 5.999000e-03f, 6.206000e-03f,
                6.413000e-03f, 6.620000e-03f, 6.827000e-03f, 7.034000e-03f, 7.241000e-03f,
                5.747000e-03f, 5.948000e-03f, 6.149000e-03f, 6.350000e-03f, 6.551000e-03f,
                6.752000e-03f, 6.953000e-03f, 5.495000e-03f, 5.690000e-03f, 5.885000e-03f,
                6.080000e-03f, 6.275000e-03f, 6.470000e-03f, 6.665000e-03f, 5.243000e-03f,
                5.432000e-03f, 5.621000e-03f, 5.810000e-03f, 5.999000e-03f, 6.188000e-03f,
                6.377000e-03f, 4.991000e-03f, 5.174000e-03f, 5.357000e-03f, 5.540000e-03f,
                5.723000e-03f, 5.906000e-03f, 6.089000e-03f, 4.739000e-03f, 4.916000e-03f,
                5.093000e-03f, 5.270000e-03f, 5.447000e-03f, 5.624000e-03f, 5.801000e-03f,
                4.487000e-03f, 4.658000e-03f, 4.829000e-03f, 5.000000e-03f, 5.171000e-03f,
                5.342000e-03f, 5.513000e-03f, 4.235000e-03f, 4.400000e-03f, 4.565000e-03f,
                4.730000e-03f, 4.895000e-03f, 5.060000e-03f, 5.225000e-03f, 8.330000e-03f,
                8.555000e-03f, 8.780000e-03f, 9.005000e-03f, 9.230000e-03f, 9.455000e-03f,
                9.680000e-03f, 8.036000e-03f, 8.255000e-03f, 8.474000e-03f, 8.693000e-03f,
                8.912000e-03f, 9.131000e-03f, 9.350000e-03f, 7.742000e-03f, 7.955000e-03f,
                8.168000e-03f, 8.381000e-03f, 8.594000e-03f, 8.807000e-03f, 9.020000e-03f,
                7.448000e-03f, 7.655000e-03f, 7.862000e-03f, 8.069000e-03f, 8.276000e-03f,
                8.483000e-03f, 8.690000e-03f, 7.154000e-03f, 7.355000e-03f, 7.556000e-03f,
                7.757000e-03f, 7.958000e-03f, 8.159000e-03f, 8.360000e-03f, 6.860000e-03f,
                7.055000e-03f, 7.250000e-03f, 7.445000e-03f, 7.640000e-03f, 7.835000e-03f,
                8.030000e-03f, 6.566000e-03f, 6.755000e-03f, 6.944000e-03f, 7.133000e-03f,
                7.322000e-03f, 7.511000e-03f, 7.700000e-03f, 6.272000e-03f, 6.455000e-03f,
                6.638000e-03f, 6.821000e-03f, 7.004000e-03f, 7.187000e-03f, 7.370000e-03f,
                5.978000e-03f, 6.155000e-03f, 6.332000e-03f, 6.509000e-03f, 6.686000e-03f,
                6.863000e-03f, 7.040000e-03f, 5.684000e-03f, 5.855000e-03f, 6.026000e-03f,
                6.197000e-03f, 6.368000e-03f, 6.539000e-03f, 6.710000e-03f, 5.390000e-03f,
                5.555000e-03f, 5.720000e-03f, 5.885000e-03f, 6.050000e-03f, 6.215000e-03f,
                6.380000e-03f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int stride in new int[] { 1, 2, 3 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = inwidth - kwidth + 1;

                                    float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] gyval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map1D x = new Map1D(inchannels, inwidth, batch, xval);
                                    Map1D gy = new Map1D(outchannels, outwidth, batch, gyval);

                                    Filter1D gw = Reference(x, gy, kwidth, stride);
                                    Filter1D gw_optimized = OptimizedReference(x, gy, kwidth, stride);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_optimized.ToArray();

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");
                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
