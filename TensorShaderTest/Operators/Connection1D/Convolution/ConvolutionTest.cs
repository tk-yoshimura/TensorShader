using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class ConvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int stride in new int[] { 1 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    int outwidth = (inwidth - kwidth) / stride + 1;

                                    float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map1D x = new Map1D(inchannels, inwidth, batch, xval);
                                    Filter1D w = new Filter1D(inchannels, outchannels, kwidth, wval);

                                    Map1D y = Reference(x, w, kwidth, stride);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch));

                                    Convolution ope = new Convolution(inwidth, inchannels, outchannels, kwidth, stride, batch);
                                    
                                    ope.Execute(x_tensor, w_tensor, y_tensor);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

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
            int inwidth = 512, inchannels = 31, outchannels = 31, ksize = 3, stride = 1;
            int outwidth = (inwidth - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            Convolution ope = new Convolution(inwidth, inchannels, outchannels, ksize, stride);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/convolution1d_trans_v2.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D x, Filter1D w, int kwidth, int stride) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, outw = (inw - kwidth) / stride + 1;

            Map1D y = new Map1D(outchannels, outw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            double sum = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                sum += x[inch, kx + ox * stride, th] * w[inch, outch, kx];
                            }

                            y[outch, ox, th] = sum;
                        }
                    }
                }
            }

            return y;
        }

        public static Map1D OptimizedReference(Map1D x, Filter1D w, int kwidth, int stride) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, outw = (inw - kwidth) / stride + 1;

            Map1D y = new Map1D(outchannels, outw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                int inmap_offset = kx * inchannels;
                int kernel_offset = kx * inchannels * outchannels;

                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        int inmap_org = inmap_offset + ox * inchannels * stride + th * inw * inchannels;
                        int outmap_idx = ox * outchannels + th * outw * outchannels;
                        int kernel_idx = kernel_offset;

                        for (int outch = 0; outch < outchannels; outch++) {
                            double sum = y[outmap_idx];

                            int inmap_idx = inmap_org;

                            for (int inch = 0; inch < inchannels; inch++) {
                                sum += x[inmap_idx] * w[kernel_idx];

                                inmap_idx++;
                                kernel_idx++;
                            }

                            y[outmap_idx] = sum;

                            outmap_idx++;
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, stride = 2, inwidth = 13, batch = 2;
            int outwidth = (inwidth - kwidth) / stride + 1;

            float[] xval = (new float[batch * inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new Map1D(inchannels, inwidth, batch, xval);
            Filter1D w = new Filter1D(inchannels, outchannels, kwidth, wval);

            Map1D y = Reference(x, w, kwidth, stride);

            float[] y_expect = {
                2.387000e-02f, 2.240000e-02f, 2.093000e-02f, 1.946000e-02f, 1.799000e-02f,
                1.652000e-02f, 1.505000e-02f, 1.358000e-02f, 1.211000e-02f, 1.064000e-02f,
                9.170000e-03f, 6.797000e-02f, 6.444200e-02f, 6.091400e-02f, 5.738600e-02f,
                5.385800e-02f, 5.033000e-02f, 4.680200e-02f, 4.327400e-02f, 3.974600e-02f,
                3.621800e-02f, 3.269000e-02f, 1.120700e-01f, 1.064840e-01f, 1.008980e-01f,
                9.531200e-02f, 8.972600e-02f, 8.414000e-02f, 7.855400e-02f, 7.296800e-02f,
                6.738200e-02f, 6.179600e-02f, 5.621000e-02f, 1.561700e-01f, 1.485260e-01f,
                1.408820e-01f, 1.332380e-01f, 1.255940e-01f, 1.179500e-01f, 1.103060e-01f,
                1.026620e-01f, 9.501800e-02f, 8.737400e-02f, 7.973000e-02f, 2.002700e-01f,
                1.905680e-01f, 1.808660e-01f, 1.711640e-01f, 1.614620e-01f, 1.517600e-01f,
                1.420580e-01f, 1.323560e-01f, 1.226540e-01f, 1.129520e-01f, 1.032500e-01f,
                2.443700e-01f, 2.326100e-01f, 2.208500e-01f, 2.090900e-01f, 1.973300e-01f,
                1.855700e-01f, 1.738100e-01f, 1.620500e-01f, 1.502900e-01f, 1.385300e-01f,
                1.267700e-01f, 3.105200e-01f, 2.956730e-01f, 2.808260e-01f, 2.659790e-01f,
                2.511320e-01f, 2.362850e-01f, 2.214380e-01f, 2.065910e-01f, 1.917440e-01f,
                1.768970e-01f, 1.620500e-01f, 3.546200e-01f, 3.377150e-01f, 3.208100e-01f,
                3.039050e-01f, 2.870000e-01f, 2.700950e-01f, 2.531900e-01f, 2.362850e-01f,
                2.193800e-01f, 2.024750e-01f, 1.855700e-01f, 3.987200e-01f, 3.797570e-01f,
                3.607940e-01f, 3.418310e-01f, 3.228680e-01f, 3.039050e-01f, 2.849420e-01f,
                2.659790e-01f, 2.470160e-01f, 2.280530e-01f, 2.090900e-01f, 4.428200e-01f,
                4.217990e-01f, 4.007780e-01f, 3.797570e-01f, 3.587360e-01f, 3.377150e-01f,
                3.166940e-01f, 2.956730e-01f, 2.746520e-01f, 2.536310e-01f, 2.326100e-01f,
                4.869200e-01f, 4.638410e-01f, 4.407620e-01f, 4.176830e-01f, 3.946040e-01f,
                3.715250e-01f, 3.484460e-01f, 3.253670e-01f, 3.022880e-01f, 2.792090e-01f,
                2.561300e-01f, 5.310200e-01f, 5.058830e-01f, 4.807460e-01f, 4.556090e-01f,
                4.304720e-01f, 4.053350e-01f, 3.801980e-01f, 3.550610e-01f, 3.299240e-01f,
                3.047870e-01f, 2.796500e-01f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");
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
                                    int outwidth = (inwidth - kwidth) / stride + 1;

                                    float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map1D x = new Map1D(inchannels, inwidth, batch, xval);
                                    Filter1D w = new Filter1D(inchannels, outchannels, kwidth, wval);

                                    Map1D y = Reference(x, w, kwidth, stride);
                                    Map1D y_optimized = OptimizedReference(x, w, kwidth, stride);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_optimized.ToArray();

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

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
