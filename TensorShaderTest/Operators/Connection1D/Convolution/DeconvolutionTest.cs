using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection1D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection1D {
    [TestClass]
    public class DeconvolutionTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 1024 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 1024 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int stride in new int[] { 1 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17 }) {
                                    int outwidth = (inwidth - kwidth) / stride + 1;

                                    float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-4f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => idx * 1e-4f).Reverse().ToArray();

                                    Map1D y = new Map1D(outchannels, outwidth, batch, yval);
                                    Filter1D w = new Filter1D(inchannels, outchannels, kwidth, wval);

                                    Map1D x = Reference(y, w, inwidth, kwidth, stride);

                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);
                                    OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch));

                                    Deconvolution ope = new Deconvolution(inwidth, outchannels, inchannels, kwidth, stride, batch);

                                    ope.Execute(y_tensor, w_tensor, x_tensor);

                                    float[] x_expect = x.ToArray();
                                    float[] x_actual = x_tensor.State;

                                    CollectionAssert.AreEqual(yval, y_tensor.State);
                                    CollectionAssert.AreEqual(wval, w_tensor.State);

                                    AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{kwidth},{stride},{inwidth},{batch}");

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

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels, ksize));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));

            Deconvolution ope = new Deconvolution(inwidth, outchannels, inchannels, ksize, stride);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/deconvolution1d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Cuda.Profiler.Stop();
        }

        public static Map1D Reference(Map1D y, Filter1D w, int inw, int kwidth, int stride) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = (inw - kwidth) / stride + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            Map1D x = new Map1D(inchannels, inw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        for (int outch = 0; outch < outchannels; outch++) {
                            double v = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                x[inch, kx + ox * stride, th] += v * w[inch, outch, kx];
                            }
                        }
                    }
                }
            }

            return x;
        }

        public static Map1D Reference2(Map1D y, Filter1D w, int inw, int kwidth, int stride) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = (inw - kwidth) / stride + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            Map1D x = new Map1D(inchannels, inw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        int inch;
                        double[] temp = new double[4];

                        for (inch = 0; inch < inchannels - inchannels % 4; inch += 4) {
                            for (int i = 0; i < 4; i++) {
                                temp[i] = x[inch + i, kx + ox * stride, th];
                            }

                            for (int outch = 0; outch < outchannels; outch++) {
                                double yv = y[outch, ox, th];

                                for (int i = 0; i < 4; i++) {
                                    temp[i] += yv * w[inch + i, outch, kx];
                                }
                            }

                            for (int i = 0; i < 4; i++) {
                                x[inch + i, kx + ox * stride, th] = temp[i];
                            }
                        }

                        if (inchannels % 4 != 0) {
                            int sets = inchannels % 4;

                            for (int i = 0; i < sets; i++) {
                                temp[i] = x[inch + i, kx + ox * stride, th];
                            }

                            for (int outch = 0; outch < outchannels; outch++) {
                                double yv = y[outch, ox, th];

                                for (int i = 0; i < sets; i++) {
                                    temp[i] += yv * w[inch + i, outch, kx];
                                }
                            }

                            for (int i = 0; i < sets; i++) {
                                x[inch + i, kx + ox * stride, th] = temp[i];
                            }
                        }
                    }
                }
            }

            return x;
        }

        public static Map1D OptimizedReference(Map1D y, Filter1D w, int inw, int kwidth) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw - kwidth + 1;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            Map1D x = new Map1D(inchannels, inw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox = 0; ox < inw; ox++) {
                     for (int inch = 0; inch < inchannels; inch++) {
                    
                        double v = 0;

                        for (int kx = 0; kx < kwidth; kx++) {
                            int ix = ox + kx - (kwidth - 1);

                            if(ix < 0 || ix >= outw) { 
                                continue;
                            }

                            for (int outch = 0; outch < outchannels; outch++) {
                                v += y[outch, ix, th] * w[inch, outch, (kwidth - 1) - kx];
                            }

                        }

                        x[inch, ox, th] = v;
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, stride = 2, inwidth = 13, batch = 2;
            int outwidth = (inwidth - kwidth) / stride + 1;

            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D y = new Map1D(outchannels, outwidth, batch, yval);
            Filter1D w = new Filter1D(inchannels, outchannels, kwidth, wval);

            Map1D x = Reference(y, w, inwidth, kwidth, stride);

            float[] x_expect = {
                9.955000e-03f, 9.900000e-03f, 9.845000e-03f, 9.790000e-03f, 9.735000e-03f,
                9.680000e-03f, 9.625000e-03f, 5.720000e-03f, 5.665000e-03f, 5.610000e-03f,
                5.555000e-03f, 5.500000e-03f, 5.445000e-03f, 5.390000e-03f, 3.503500e-02f,
                3.480400e-02f, 3.457300e-02f, 3.434200e-02f, 3.411100e-02f, 3.388000e-02f,
                3.364900e-02f, 1.999800e-02f, 1.982200e-02f, 1.964600e-02f, 1.947000e-02f,
                1.929400e-02f, 1.911800e-02f, 1.894200e-02f, 6.359100e-02f, 6.311800e-02f,
                6.264500e-02f, 6.217200e-02f, 6.169900e-02f, 6.122600e-02f, 6.075300e-02f,
                3.427600e-02f, 3.397900e-02f, 3.368200e-02f, 3.338500e-02f, 3.308800e-02f,
                3.279100e-02f, 3.249400e-02f, 9.214700e-02f, 9.143200e-02f, 9.071700e-02f,
                9.000200e-02f, 8.928700e-02f, 8.857200e-02f, 8.785700e-02f, 4.855400e-02f,
                4.813600e-02f, 4.771800e-02f, 4.730000e-02f, 4.688200e-02f, 4.646400e-02f,
                4.604600e-02f, 1.207030e-01f, 1.197460e-01f, 1.187890e-01f, 1.178320e-01f,
                1.168750e-01f, 1.159180e-01f, 1.149610e-01f, 6.283200e-02f, 6.229300e-02f,
                6.175400e-02f, 6.121500e-02f, 6.067600e-02f, 6.013700e-02f, 5.959800e-02f,
                1.492590e-01f, 1.480600e-01f, 1.468610e-01f, 1.456620e-01f, 1.444630e-01f,
                1.432640e-01f, 1.420650e-01f, 7.711000e-02f, 7.645000e-02f, 7.579000e-02f,
                7.513000e-02f, 7.447000e-02f, 7.381000e-02f, 7.315000e-02f, 2.629000e-02f,
                2.563000e-02f, 2.497000e-02f, 2.431000e-02f, 2.365000e-02f, 2.299000e-02f,
                2.233000e-02f, 1.515250e-01f, 1.507440e-01f, 1.499630e-01f, 1.491820e-01f,
                1.484010e-01f, 1.476200e-01f, 1.468390e-01f, 9.138800e-02f, 9.060700e-02f,
                8.982600e-02f, 8.904500e-02f, 8.826400e-02f, 8.748300e-02f, 8.670200e-02f,
                2.063710e-01f, 2.046880e-01f, 2.030050e-01f, 2.013220e-01f, 1.996390e-01f,
                1.979560e-01f, 1.962730e-01f, 1.056660e-01f, 1.047640e-01f, 1.038620e-01f,
                1.029600e-01f, 1.020580e-01f, 1.011560e-01f, 1.002540e-01f, 2.349270e-01f,
                2.330020e-01f, 2.310770e-01f, 2.291520e-01f, 2.272270e-01f, 2.253020e-01f,
                2.233770e-01f, 1.199440e-01f, 1.189210e-01f, 1.178980e-01f, 1.168750e-01f,
                1.158520e-01f, 1.148290e-01f, 1.138060e-01f, 2.634830e-01f, 2.613160e-01f,
                2.591490e-01f, 2.569820e-01f, 2.548150e-01f, 2.526480e-01f, 2.504810e-01f,
                1.342220e-01f, 1.330780e-01f, 1.319340e-01f, 1.307900e-01f, 1.296460e-01f,
                1.285020e-01f, 1.273580e-01f, 2.920390e-01f, 2.896300e-01f, 2.872210e-01f,
                2.848120e-01f, 2.824030e-01f, 2.799940e-01f, 2.775850e-01f, 1.485000e-01f,
                1.472350e-01f, 1.459700e-01f, 1.447050e-01f, 1.434400e-01f, 1.421750e-01f,
                1.409100e-01f, 3.205950e-01f, 3.179440e-01f, 3.152930e-01f, 3.126420e-01f,
                3.099910e-01f, 3.073400e-01f, 3.046890e-01f, 1.627780e-01f, 1.613920e-01f,
                1.600060e-01f, 1.586200e-01f, 1.572340e-01f, 1.558480e-01f, 1.544620e-01f,
                5.605600e-02f, 5.467000e-02f, 5.328400e-02f, 5.189800e-02f, 5.051200e-02f,
                4.912600e-02f, 4.774000e-02f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{kwidth},{stride},{inwidth},{batch}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int stride in new int[] { 1 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = (inwidth - kwidth) / stride + 1;

                                    float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-4f).ToArray();
                                    float[] wval = (new float[kwidth * inchannels * outchannels]).Select((_, idx) => idx * 1e-4f).Reverse().ToArray();

                                    Map1D y = new Map1D(outchannels, outwidth, batch, yval);
                                    Filter1D w = new Filter1D(inchannels, outchannels, kwidth, wval);

                                    Map1D x = Reference(y, w, inwidth, kwidth, stride);
                                    Map1D x_optimized = OptimizedReference(y, w, inwidth, kwidth);

                                    float[] x_expect = x.ToArray();
                                    float[] x_actual = x_optimized.ToArray();

                                    AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

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
