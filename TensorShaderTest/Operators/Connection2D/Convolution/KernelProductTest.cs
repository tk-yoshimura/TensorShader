using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection2D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class KernelProductTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] gyval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map2D x = new(inchannels, inwidth, inheight, batch, xval);
                                    Map2D gy = new(outchannels, outwidth, outheight, batch, gyval);

                                    Filter2D gw = Reference(x, gy, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                    OverflowCheckedTensor gy_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), gyval);

                                    OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight));

                                    KernelProduct ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, batch);

                                    ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_tensor.State.Value;

                                    CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                    CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void ExecuteFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] gyval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map2D x = new(inchannels, inwidth, inheight, batch, xval);
                                    Map2D gy = new(outchannels, outwidth, outheight, batch, gyval);

                                    Filter2D gw = Reference(x, gy, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                    OverflowCheckedTensor gy_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), gyval);

                                    OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight));

                                    KernelProduct ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, batch);

                                    ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_tensor.State.Value;

                                    CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                    CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void ExecuteCudnnTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = true;

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach (int kheight in new int[] { 1, 3, 5 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int inheight in new int[] { kheight, kheight * 2, 8, 9, 13, 17, 25 }) {
                                foreach (int inwidth in new int[] { kwidth, kwidth * 2, 8, 9, 13, 17, 25 }) {
                                    int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

                                    float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] gyval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    Map2D x = new(inchannels, inwidth, inheight, batch, xval);
                                    Map2D gy = new(outchannels, outwidth, outheight, batch, gyval);

                                    Filter2D gw = Reference(x, gy, kwidth, kheight);

                                    OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                    OverflowCheckedTensor gy_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), gyval);

                                    OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight));

                                    KernelProduct ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, batch);

                                    ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_tensor.State.Value;

                                    CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                                    CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            float max_err = 0;

            Random random = new(1234);

            int batch = 3;
            int inchannels = 49, outchannels = 50;
            int kwidth = 5, kheight = 3;
            int inwidth = 250, inheight = 196;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map2D x = new(inchannels, inwidth, inheight, batch, xval);
            Map2D gy = new(outchannels, outwidth, outheight, batch, gyval);

            Filter2D gw = Reference(x, gy, kwidth, kheight);

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
            OverflowCheckedTensor gy_tensor = new(Shape.Map2D(outchannels, outwidth, outheight, batch), gyval);

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight));

            KernelProduct ope = new(inwidth, inheight, inchannels, outchannels, kwidth, kheight, batch);

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor gy_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            KernelProduct ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_2d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor gy_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            KernelProduct ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_2d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedCudnnTest() {
            if (!TensorShaderCudaBackend.Environment.CudnnExists) {
                Console.WriteLine("test was skipped. Cudnn library not exists.");
                Assert.Inconclusive();
            }

            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = true;

            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor gy_tensor = new(Shape.Map2D(outchannels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            KernelProduct ope = new(inwidth, inheight, inchannels, outchannels, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_2d_cudnn.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter2D Reference(Map2D x, Map2D gy, int kwidth, int kheight) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter2D w = new(inchannels, outchannels, kwidth, kheight);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ix, iy = ky, ox, oy = 0; oy < outh; iy++, oy++) {
                            for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                for (int inch, outch = 0; outch < outchannels; outch++) {
                                    for (inch = 0; inch < inchannels; inch++) {
                                        w[inch, outch, kx, ky] += x[inch, ix, iy, th] * gy[outch, ox, oy, th];
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
            int inchannels = 7, outchannels = 11, kwidth = 3, kheight = 5, inwidth = 13, inheight = 17;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new(inchannels, inwidth, inheight, 1, xval);
            Map2D gy = new(outchannels, outwidth, outheight, 1, gyval);

            Filter2D gw = Reference(x, gy, kwidth, kheight);

            float[] gw_expect = {
                4.356452179e+01f, 4.367764282e+01f, 4.379074478e+01f, 4.390384674e+01f, 4.401698303e+01f, 4.413009262e+01f, 4.424319839e+01f,
                4.348141861e+01f, 4.359441757e+01f, 4.370737839e+01f, 4.382035065e+01f, 4.393332291e+01f, 4.404628754e+01f, 4.415926743e+01f,
                4.339836121e+01f, 4.351118088e+01f, 4.362400818e+01f, 4.373684692e+01f, 4.384966660e+01f, 4.396248627e+01f, 4.407532501e+01f,
                4.331527710e+01f, 4.342795944e+01f, 4.354064941e+01f, 4.365332794e+01f, 4.376600647e+01f, 4.387868500e+01f, 4.399139023e+01f,
                4.323217773e+01f, 4.334473801e+01f, 4.345727539e+01f, 4.356982422e+01f, 4.368236923e+01f, 4.379490280e+01f, 4.390743637e+01f,
                4.314910126e+01f, 4.326149750e+01f, 4.337390518e+01f, 4.348630142e+01f, 4.359870148e+01f, 4.371110916e+01f, 4.382349014e+01f,
                4.306602859e+01f, 4.317828369e+01f, 4.329055023e+01f, 4.340278244e+01f, 4.351504898e+01f, 4.362729645e+01f, 4.373954773e+01f,
                4.298295975e+01f, 4.309506607e+01f, 4.320716095e+01f, 4.331927490e+01f, 4.343139648e+01f, 4.354350662e+01f, 4.365562820e+01f,
                4.289986801e+01f, 4.301183319e+01f, 4.312380981e+01f, 4.323575974e+01f, 4.334774017e+01f, 4.345971298e+01f, 4.357167816e+01f,
                4.281676865e+01f, 4.292860031e+01f, 4.304044342e+01f, 4.315225601e+01f, 4.326406479e+01f, 4.337590790e+01f, 4.348772812e+01f,
                4.273370361e+01f, 4.284536362e+01f, 4.295704651e+01f, 4.306874084e+01f, 4.318040466e+01f, 4.329211044e+01f, 4.340378952e+01f,
                4.435631943e+01f, 4.446943665e+01f, 4.458254242e+01f, 4.469565201e+01f, 4.480876923e+01f, 4.492188644e+01f, 4.503499222e+01f,
                4.427222061e+01f, 4.438518906e+01f, 4.449818420e+01f, 4.461114502e+01f, 4.472409821e+01f, 4.483708191e+01f, 4.495003891e+01f,
                4.418814850e+01f, 4.430099106e+01f, 4.441380692e+01f, 4.452664566e+01f, 4.463946533e+01f, 4.475228119e+01f, 4.486511612e+01f,
                4.410407257e+01f, 4.421673965e+01f, 4.432942963e+01f, 4.444211960e+01f, 4.455479813e+01f, 4.466748810e+01f, 4.478015518e+01f,
                4.401997757e+01f, 4.413252640e+01f, 4.424507523e+01f, 4.435760880e+01f, 4.447016144e+01f, 4.458269501e+01f, 4.469522858e+01f,
                4.393589020e+01f, 4.404831314e+01f, 4.416069031e+01f, 4.427309036e+01f, 4.438548279e+01f, 4.449789047e+01f, 4.461029816e+01f,
                4.385180664e+01f, 4.396407318e+01f, 4.407632446e+01f, 4.418858719e+01f, 4.430083466e+01f, 4.441307068e+01f, 4.452533722e+01f,
                4.376772690e+01f, 4.387984085e+01f, 4.399194336e+01f, 4.410404968e+01f, 4.421617889e+01f, 4.432827759e+01f, 4.444040298e+01f,
                4.368365860e+01f, 4.379561234e+01f, 4.390757751e+01f, 4.401955795e+01f, 4.413151550e+01f, 4.424349594e+01f, 4.435546112e+01f,
                4.359955978e+01f, 4.371138763e+01f, 4.382320786e+01f, 4.393503571e+01f, 4.404686356e+01f, 4.415870285e+01f, 4.427051544e+01f,
                4.351548386e+01f, 4.362715912e+01f, 4.373884201e+01f, 4.385051727e+01f, 4.396220779e+01f, 4.407389069e+01f, 4.418558884e+01f,
                4.514813232e+01f, 4.526122284e+01f, 4.537432861e+01f, 4.548744202e+01f, 4.560055542e+01f, 4.571367264e+01f, 4.582676697e+01f,
                4.506301880e+01f, 4.517599106e+01f, 4.528895950e+01f, 4.540193939e+01f, 4.551490402e+01f, 4.562786102e+01f, 4.574084854e+01f,
                4.497793579e+01f, 4.509077072e+01f, 4.520359421e+01f, 4.531641388e+01f, 4.542924500e+01f, 4.554205704e+01f, 4.565491104e+01f,
                4.489285660e+01f, 4.500555038e+01f, 4.511822510e+01f, 4.523088837e+01f, 4.534359360e+01f, 4.545626450e+01f, 4.556895828e+01f,
                4.480775833e+01f, 4.492029953e+01f, 4.503285599e+01f, 4.514537811e+01f, 4.525792694e+01f, 4.537046814e+01f, 4.548301315e+01f,
                4.472269058e+01f, 4.483508682e+01f, 4.494747925e+01f, 4.505989456e+01f, 4.517226791e+01f, 4.528466415e+01f, 4.539706802e+01f,
                4.463759613e+01f, 4.474986267e+01f, 4.486212158e+01f, 4.497435379e+01f, 4.508660889e+01f, 4.519888306e+01f, 4.531114197e+01f,
                4.455250549e+01f, 4.466463470e+01f, 4.477674103e+01f, 4.488885498e+01f, 4.500094986e+01f, 4.511307907e+01f, 4.522519302e+01f,
                4.446741104e+01f, 4.457938004e+01f, 4.469136810e+01f, 4.480334854e+01f, 4.491531372e+01f, 4.502725983e+01f, 4.513925171e+01f,
                4.438233185e+01f, 4.449417496e+01f, 4.460597992e+01f, 4.471782303e+01f, 4.482965088e+01f, 4.494146729e+01f, 4.505329514e+01f,
                4.429726028e+01f, 4.440894318e+01f, 4.452062225e+01f, 4.463231277e+01f, 4.474399567e+01f, 4.485568619e+01f, 4.496735764e+01f,
                5.385780334e+01f, 5.397091293e+01f, 5.408403778e+01f, 5.419713593e+01f, 5.431024551e+01f, 5.442338562e+01f, 5.453649139e+01f,
                5.376171494e+01f, 5.387468719e+01f, 5.398766327e+01f, 5.410062027e+01f, 5.421359634e+01f, 5.432656097e+01f, 5.443952179e+01f,
                5.366559601e+01f, 5.377846146e+01f, 5.389127731e+01f, 5.400410461e+01f, 5.411691284e+01f, 5.422975540e+01f, 5.434257126e+01f,
                5.356952667e+01f, 5.368219757e+01f, 5.379487991e+01f, 5.390757751e+01f, 5.402025986e+01f, 5.413293076e+01f, 5.424563599e+01f,
                5.347344208e+01f, 5.358597183e+01f, 5.369850922e+01f, 5.381105042e+01f, 5.392358017e+01f, 5.403615570e+01f, 5.414868546e+01f,
                5.337731934e+01f, 5.348973083e+01f, 5.360213089e+01f, 5.371452713e+01f, 5.382691956e+01f, 5.393931198e+01f, 5.405171204e+01f,
                5.328123474e+01f, 5.339348602e+01f, 5.350575638e+01f, 5.361800385e+01f, 5.373026276e+01f, 5.384250641e+01f, 5.395476532e+01f,
                5.318513107e+01f, 5.329726028e+01f, 5.340935516e+01f, 5.352147675e+01f, 5.363358307e+01f, 5.374570084e+01f, 5.385779953e+01f,
                5.308904648e+01f, 5.320101929e+01f, 5.331297684e+01f, 5.342493439e+01f, 5.353689957e+01f, 5.364887238e+01f, 5.376086044e+01f,
                5.299293518e+01f, 5.310475540e+01f, 5.321659470e+01f, 5.332841492e+01f, 5.344026184e+01f, 5.355207062e+01f, 5.366390228e+01f,
                5.289685059e+01f, 5.300852203e+01f, 5.312021255e+01f, 5.323189545e+01f, 5.334359741e+01f, 5.345527267e+01f, 5.356697083e+01f,
                5.464960480e+01f, 5.476271820e+01f, 5.487582397e+01f, 5.498892975e+01f, 5.510205078e+01f, 5.521516037e+01f, 5.532828903e+01f,
                5.455249786e+01f, 5.466547775e+01f, 5.477844620e+01f, 5.489141846e+01f, 5.500437546e+01f, 5.511735153e+01f, 5.523033142e+01f,
                5.445539856e+01f, 5.456823349e+01f, 5.468106079e+01f, 5.479390335e+01f, 5.490672302e+01f, 5.501952362e+01f, 5.513237762e+01f,
                5.435829926e+01f, 5.447099304e+01f, 5.458368301e+01f, 5.469636536e+01f, 5.480903625e+01f, 5.492173386e+01f, 5.503439331e+01f,
                5.426121140e+01f, 5.437376022e+01f, 5.448630142e+01f, 5.459883118e+01f, 5.471138763e+01f, 5.482392883e+01f, 5.493644714e+01f,
                5.416411591e+01f, 5.427651978e+01f, 5.438889694e+01f, 5.450131226e+01f, 5.461368561e+01f, 5.472608185e+01f, 5.483850098e+01f,
                5.406700897e+01f, 5.417926407e+01f, 5.429151154e+01f, 5.440378189e+01f, 5.451602936e+01f, 5.462829208e+01f, 5.474053955e+01f,
                5.396993637e+01f, 5.408202744e+01f, 5.419415283e+01f, 5.430625916e+01f, 5.441836548e+01f, 5.453047943e+01f, 5.464258957e+01f,
                5.387282944e+01f, 5.398480606e+01f, 5.409676743e+01f, 5.420873260e+01f, 5.432067871e+01f, 5.443266678e+01f, 5.454463577e+01f,
                5.377574158e+01f, 5.388756943e+01f, 5.399940109e+01f, 5.411120605e+01f, 5.422303009e+01f, 5.433484650e+01f, 5.444669342e+01f,
                5.367863464e+01f, 5.379031754e+01f, 5.390200043e+01f, 5.401368332e+01f, 5.412536621e+01f, 5.423706436e+01f, 5.434872818e+01f,
                5.544138336e+01f, 5.555449295e+01f, 5.566762543e+01f, 5.578073120e+01f, 5.589382935e+01f, 5.600696945e+01f, 5.612007523e+01f,
                5.534329605e+01f, 5.545627975e+01f, 5.556924057e+01f, 5.568219376e+01f, 5.579517746e+01f, 5.590814590e+01f, 5.602110672e+01f,
                5.524519348e+01f, 5.535801315e+01f, 5.547084808e+01f, 5.558369827e+01f, 5.569650650e+01f, 5.580931473e+01f, 5.592217255e+01f,
                5.514709854e+01f, 5.525976944e+01f, 5.537246323e+01f, 5.548516464e+01f, 5.559781647e+01f, 5.571051025e+01f, 5.582321167e+01f,
                5.504899597e+01f, 5.516154099e+01f, 5.527408600e+01f, 5.538662338e+01f, 5.549916458e+01f, 5.561169434e+01f, 5.572423172e+01f,
                5.495088959e+01f, 5.506328964e+01f, 5.517570496e+01f, 5.528809738e+01f, 5.540048218e+01f, 5.551288986e+01f, 5.562528992e+01f,
                5.485280609e+01f, 5.496505737e+01f, 5.507730484e+01f, 5.518956375e+01f, 5.530181122e+01f, 5.541409683e+01f, 5.552633286e+01f,
                5.475469589e+01f, 5.486680222e+01f, 5.497892761e+01f, 5.509103012e+01f, 5.520315170e+01f, 5.531526184e+01f, 5.542737961e+01f,
                5.465661621e+01f, 5.476857376e+01f, 5.488055038e+01f, 5.499251938e+01f, 5.510449219e+01f, 5.521645355e+01f, 5.532842636e+01f,
                5.455850601e+01f, 5.467033386e+01f, 5.478215790e+01f, 5.489397049e+01f, 5.500580215e+01f, 5.511763763e+01f, 5.522948074e+01f,
                5.446041107e+01f, 5.457209778e+01f, 5.468377304e+01f, 5.479547119e+01f, 5.490713882e+01f, 5.501884079e+01f, 5.513052368e+01f,
                6.415110779e+01f, 6.426421356e+01f, 6.437731934e+01f, 6.449043274e+01f, 6.460353851e+01f, 6.471664429e+01f, 6.482979584e+01f,
                6.404198456e+01f, 6.415496826e+01f, 6.426792145e+01f, 6.438089752e+01f, 6.449383545e+01f, 6.460684204e+01f, 6.471981049e+01f,
                6.393289185e+01f, 6.404569244e+01f, 6.415852356e+01f, 6.427134705e+01f, 6.438417816e+01f, 6.449701691e+01f, 6.460984802e+01f,
                6.382375336e+01f, 6.393644714e+01f, 6.404915619e+01f, 6.416181183e+01f, 6.427450562e+01f, 6.438717651e+01f, 6.449986267e+01f,
                6.371466064e+01f, 6.382720184e+01f, 6.393973160e+01f, 6.405227661e+01f, 6.416483307e+01f, 6.427735138e+01f, 6.438991547e+01f,
                6.360554123e+01f, 6.371795273e+01f, 6.383034897e+01f, 6.394272232e+01f, 6.405513000e+01f, 6.416753387e+01f, 6.427994537e+01f,
                6.349644089e+01f, 6.360871506e+01f, 6.372094345e+01f, 6.383318710e+01f, 6.394545746e+01f, 6.405770874e+01f, 6.416996002e+01f,
                6.338731766e+01f, 6.349943542e+01f, 6.361155701e+01f, 6.372366714e+01f, 6.383578491e+01f, 6.394790649e+01f, 6.406000519e+01f,
                6.327820587e+01f, 6.339018631e+01f, 6.350217056e+01f, 6.361413956e+01f, 6.372608185e+01f, 6.383806229e+01f, 6.395003891e+01f,
                6.316911697e+01f, 6.328094482e+01f, 6.339276123e+01f, 6.350459671e+01f, 6.361641312e+01f, 6.372824860e+01f, 6.384006500e+01f,
                6.305998611e+01f, 6.317168427e+01f, 6.328339005e+01f, 6.339505386e+01f, 6.350674057e+01f, 6.361842346e+01f, 6.373010635e+01f,
                6.494288635e+01f, 6.505597687e+01f, 6.516909790e+01f, 6.528220367e+01f, 6.539535522e+01f, 6.550844574e+01f, 6.562155914e+01f,
                6.483277130e+01f, 6.494576263e+01f, 6.505871582e+01f, 6.517166901e+01f, 6.528466034e+01f, 6.539762115e+01f, 6.551060486e+01f,
                6.472264862e+01f, 6.483550262e+01f, 6.494833374e+01f, 6.506114197e+01f, 6.517395782e+01f, 6.528680420e+01f, 6.539961243e+01f,
                6.461255646e+01f, 6.472522736e+01f, 6.483794403e+01f, 6.495062256e+01f, 6.506330109e+01f, 6.517595673e+01f, 6.528866577e+01f,
                6.450244141e+01f, 6.461500549e+01f, 6.472753143e+01f, 6.484006500e+01f, 6.495261383e+01f, 6.506513214e+01f, 6.517767334e+01f,
                6.439234161e+01f, 6.450473022e+01f, 6.461714172e+01f, 6.472951508e+01f, 6.484194183e+01f, 6.495432281e+01f, 6.506672668e+01f,
                6.428222656e+01f, 6.439447021e+01f, 6.450672913e+01f, 6.461897278e+01f, 6.473125458e+01f, 6.484348297e+01f, 6.495574188e+01f,
                6.417211151e+01f, 6.428421783e+01f, 6.439633942e+01f, 6.450843811e+01f, 6.462055206e+01f, 6.473266602e+01f, 6.484478760e+01f,
                6.406201935e+01f, 6.417398071e+01f, 6.428595734e+01f, 6.439788818e+01f, 6.450988770e+01f, 6.462185669e+01f, 6.473380280e+01f,
                6.395190811e+01f, 6.406373596e+01f, 6.417553711e+01f, 6.428737640e+01f, 6.439920044e+01f, 6.451103210e+01f, 6.462285614e+01f,
                6.384178543e+01f, 6.395345688e+01f, 6.406513977e+01f, 6.417682648e+01f, 6.428852081e+01f, 6.440019989e+01f, 6.451189423e+01f,
                6.573468018e+01f, 6.584778595e+01f, 6.596092987e+01f, 6.607398987e+01f, 6.618712616e+01f, 6.630022430e+01f, 6.641335297e+01f,
                6.562356567e+01f, 6.573653412e+01f, 6.584948730e+01f, 6.596245575e+01f, 6.607542419e+01f, 6.618840790e+01f, 6.630137634e+01f,
                6.551245117e+01f, 6.562527466e+01f, 6.573810577e+01f, 6.585093689e+01f, 6.596376038e+01f, 6.607657623e+01f, 6.618939209e+01f,
                6.540134430e+01f, 6.551401520e+01f, 6.562671661e+01f, 6.573941040e+01f, 6.585209656e+01f, 6.596478271e+01f, 6.607744598e+01f,
                6.529022980e+01f, 6.540277863e+01f, 6.551531982e+01f, 6.562785339e+01f, 6.574039459e+01f, 6.585291290e+01f, 6.596550751e+01f,
                6.517909241e+01f, 6.529148102e+01f, 6.540391541e+01f, 6.551633453e+01f, 6.562870789e+01f, 6.574110413e+01f, 6.585351562e+01f,
                6.506800079e+01f, 6.518027496e+01f, 6.529250336e+01f, 6.540480042e+01f, 6.551703644e+01f, 6.562928009e+01f, 6.574152374e+01f,
                6.495690918e+01f, 6.506900024e+01f, 6.518113708e+01f, 6.529321289e+01f, 6.540532684e+01f, 6.551745605e+01f, 6.562958527e+01f,
                6.484579468e+01f, 6.495777130e+01f, 6.506971741e+01f, 6.518167877e+01f, 6.529367065e+01f, 6.540562439e+01f, 6.551760864e+01f,
                6.473468018e+01f, 6.484652710e+01f, 6.495832062e+01f, 6.507012177e+01f, 6.518196869e+01f, 6.529380798e+01f, 6.540561676e+01f,
                6.462356567e+01f, 6.473524475e+01f, 6.484693146e+01f, 6.495859528e+01f, 6.507029724e+01f, 6.518196869e+01f, 6.529368591e+01f,
                7.444440460e+01f, 7.455748749e+01f, 7.467061615e+01f, 7.478370667e+01f, 7.489686584e+01f, 7.500995636e+01f, 7.512306213e+01f,
                7.432226562e+01f, 7.443519592e+01f, 7.454817963e+01f, 7.466116333e+01f, 7.477412415e+01f, 7.488713074e+01f, 7.500005341e+01f,
                7.420013428e+01f, 7.431297302e+01f, 7.442581177e+01f, 7.453860474e+01f, 7.465142059e+01f, 7.476425934e+01f, 7.487709045e+01f,
                7.407801056e+01f, 7.419071198e+01f, 7.430339813e+01f, 7.441603851e+01f, 7.452877045e+01f, 7.464144135e+01f, 7.475411224e+01f,
                7.395592499e+01f, 7.406841278e+01f, 7.418098450e+01f, 7.429351807e+01f, 7.440608215e+01f, 7.451858521e+01f, 7.463110352e+01f,
                7.383377838e+01f, 7.394618988e+01f, 7.405855560e+01f, 7.417094421e+01f, 7.428336334e+01f, 7.439574432e+01f, 7.450814819e+01f,
                7.371163940e+01f, 7.382390594e+01f, 7.393613434e+01f, 7.404841614e+01f, 7.416062164e+01f, 7.427294159e+01f, 7.438516998e+01f,
                7.358952332e+01f, 7.370162964e+01f, 7.381375122e+01f, 7.392586517e+01f, 7.403796387e+01f, 7.415007019e+01f, 7.426220703e+01f,
                7.346743774e+01f, 7.357934570e+01f, 7.369132996e+01f, 7.380331421e+01f, 7.391526031e+01f, 7.402724457e+01f, 7.413921356e+01f,
                7.334529114e+01f, 7.345710754e+01f, 7.356894684e+01f, 7.368076324e+01f, 7.379254150e+01f, 7.390442657e+01f, 7.401625061e+01f,
                7.322315216e+01f, 7.333482361e+01f, 7.344651031e+01f, 7.355819702e+01f, 7.366987610e+01f, 7.378160095e+01f, 7.389325714e+01f,
                7.523615265e+01f, 7.534927368e+01f, 7.546238708e+01f, 7.557550812e+01f, 7.568858337e+01f, 7.580172729e+01f, 7.591485596e+01f,
                7.511306000e+01f, 7.522598267e+01f, 7.533899689e+01f, 7.545197296e+01f, 7.556491852e+01f, 7.567790222e+01f, 7.579087830e+01f,
                7.498993683e+01f, 7.510274506e+01f, 7.521558380e+01f, 7.532843781e+01f, 7.544123077e+01f, 7.555406952e+01f, 7.566690063e+01f,
                7.486677551e+01f, 7.497949219e+01f, 7.509220123e+01f, 7.520484924e+01f, 7.531754303e+01f, 7.543023682e+01f, 7.554290771e+01f,
                7.474367523e+01f, 7.485621643e+01f, 7.496875763e+01f, 7.508127594e+01f, 7.519382477e+01f, 7.530636597e+01f, 7.541892242e+01f,
                7.462056732e+01f, 7.473294830e+01f, 7.484533691e+01f, 7.495774078e+01f, 7.507015991e+01f, 7.518254852e+01f, 7.529494476e+01f,
                7.449739838e+01f, 7.460968781e+01f, 7.472193909e+01f, 7.483419800e+01f, 7.494642639e+01f, 7.505870056e+01f, 7.517093658e+01f,
                7.437432098e+01f, 7.448642731e+01f, 7.459853363e+01f, 7.471063232e+01f, 7.482276154e+01f, 7.493484497e+01f, 7.504700470e+01f,
                7.425115967e+01f, 7.436314392e+01f, 7.447512054e+01f, 7.458711243e+01f, 7.469908142e+01f, 7.481101227e+01f, 7.492301941e+01f,
                7.412806702e+01f, 7.423991394e+01f, 7.435169983e+01f, 7.446353149e+01f, 7.457537842e+01f, 7.468722534e+01f, 7.479901123e+01f,
                7.400492096e+01f, 7.411662292e+01f, 7.422829437e+01f, 7.433998871e+01f, 7.445168304e+01f, 7.456336212e+01f, 7.467503357e+01f,
                7.602797699e+01f, 7.614109802e+01f, 7.625419617e+01f, 7.636728668e+01f, 7.648042297e+01f, 7.659354401e+01f, 7.670663452e+01f,
                7.590381622e+01f, 7.601677704e+01f, 7.612977600e+01f, 7.624273682e+01f, 7.635573578e+01f, 7.646868134e+01f, 7.658164978e+01f,
                7.577969360e+01f, 7.589254761e+01f, 7.600536346e+01f, 7.611820221e+01f, 7.623101807e+01f, 7.634384918e+01f, 7.645665741e+01f,
                7.565559387e+01f, 7.576827240e+01f, 7.588092804e+01f, 7.599366760e+01f, 7.610636139e+01f, 7.621901703e+01f, 7.633170319e+01f,
                7.553149414e+01f, 7.564399719e+01f, 7.575654602e+01f, 7.586909485e+01f, 7.598162079e+01f, 7.609417725e+01f, 7.620668793e+01f,
                7.540734100e+01f, 7.551974487e+01f, 7.563214111e+01f, 7.574452209e+01f, 7.585692596e+01f, 7.596931458e+01f, 7.608175659e+01f,
                7.528322601e+01f, 7.539547729e+01f, 7.550771332e+01f, 7.562000275e+01f, 7.573226929e+01f, 7.584447479e+01f, 7.595675659e+01f,
                7.515908051e+01f, 7.527121735e+01f, 7.538332367e+01f, 7.549538422e+01f, 7.560754395e+01f, 7.571965027e+01f, 7.583175659e+01f,
                7.503494263e+01f, 7.514692688e+01f, 7.525888062e+01f, 7.537084961e+01f, 7.548285675e+01f, 7.559480286e+01f, 7.570675659e+01f,
                7.491082001e+01f, 7.502268219e+01f, 7.513449860e+01f, 7.524631500e+01f, 7.535811615e+01f, 7.546995544e+01f, 7.558181763e+01f,
                7.478670502e+01f, 7.489839935e+01f, 7.501008606e+01f, 7.512177277e+01f, 7.523344421e+01f, 7.534512329e+01f, 7.545684052e+01f,
                8.473767090e+01f, 8.485077667e+01f, 8.496390533e+01f, 8.507701874e+01f, 8.519011688e+01f, 8.530324554e+01f, 8.541634369e+01f,
                8.460253143e+01f, 8.471550751e+01f, 8.482843781e+01f, 8.494149017e+01f, 8.505440521e+01f, 8.516737366e+01f, 8.528035736e+01f,
                8.446736908e+01f, 8.458022308e+01f, 8.469306946e+01f, 8.480583954e+01f, 8.491866302e+01f, 8.503153992e+01f, 8.514434814e+01f,
                8.433224487e+01f, 8.444495392e+01f, 8.455760193e+01f, 8.467029572e+01f, 8.478300476e+01f, 8.489566803e+01f, 8.500836182e+01f,
                8.419712830e+01f, 8.430968475e+01f, 8.442219543e+01f, 8.453472900e+01f, 8.464732361e+01f, 8.475986481e+01f, 8.487236023e+01f,
                8.406198883e+01f, 8.417437744e+01f, 8.428674316e+01f, 8.439917755e+01f, 8.451157379e+01f, 8.462396240e+01f, 8.473637390e+01f,
                8.392684174e+01f, 8.403910065e+01f, 8.415134430e+01f, 8.426360321e+01f, 8.437586975e+01f, 8.448815155e+01f, 8.460036469e+01f,
                8.379173279e+01f, 8.390379333e+01f, 8.401593018e+01f, 8.412802887e+01f, 8.424012756e+01f, 8.435228729e+01f, 8.446439362e+01f,
                8.365654755e+01f, 8.376855469e+01f, 8.388053894e+01f, 8.399248505e+01f, 8.410442352e+01f, 8.421640015e+01f, 8.432840729e+01f,
                8.352146912e+01f, 8.363327026e+01f, 8.374511719e+01f, 8.385692596e+01f, 8.396872711e+01f, 8.408057404e+01f, 8.419240570e+01f,
                8.338631439e+01f, 8.349800873e+01f, 8.360966492e+01f, 8.372134399e+01f, 8.383303070e+01f, 8.394475555e+01f, 8.405644226e+01f,
                8.552946472e+01f, 8.564257050e+01f, 8.575567627e+01f, 8.586878204e+01f, 8.598191071e+01f, 8.609502411e+01f, 8.620813751e+01f,
                8.539328766e+01f, 8.550627899e+01f, 8.561925507e+01f, 8.573222351e+01f, 8.584519196e+01f, 8.595815277e+01f, 8.607111359e+01f,
                8.525717163e+01f, 8.537003326e+01f, 8.548281097e+01f, 8.559567261e+01f, 8.570848846e+01f, 8.582131195e+01f, 8.593415070e+01f,
                8.512106323e+01f, 8.523371887e+01f, 8.534638977e+01f, 8.545906830e+01f, 8.557176971e+01f, 8.568444061e+01f, 8.579715729e+01f,
                8.498490143e+01f, 8.509741974e+01f, 8.520994568e+01f, 8.532254028e+01f, 8.543506622e+01f, 8.554760742e+01f, 8.566014099e+01f,
                8.484874725e+01f, 8.496113586e+01f, 8.507357788e+01f, 8.518593597e+01f, 8.529840851e+01f, 8.541076660e+01f, 8.552314758e+01f,
                8.471264648e+01f, 8.482486725e+01f, 8.493714905e+01f, 8.504944611e+01f, 8.516164398e+01f, 8.527390289e+01f, 8.538618469e+01f,
                8.457651520e+01f, 8.468861389e+01f, 8.480071259e+01f, 8.491285706e+01f, 8.502494049e+01f, 8.513707733e+01f, 8.524919128e+01f,
                8.444034576e+01f, 8.455234528e+01f, 8.466430664e+01f, 8.477625275e+01f, 8.488825226e+01f, 8.500022125e+01f, 8.511213684e+01f,
                8.430423737e+01f, 8.441609192e+01f, 8.452785492e+01f, 8.463973236e+01f, 8.475151062e+01f, 8.486334991e+01f, 8.497520447e+01f,
                8.416809082e+01f, 8.427976227e+01f, 8.439144135e+01f, 8.450314331e+01f, 8.461483002e+01f, 8.472647858e+01f, 8.483818817e+01f,
                8.632123566e+01f, 8.643437195e+01f, 8.654747772e+01f, 8.666058350e+01f, 8.677369690e+01f, 8.688677216e+01f, 8.699989319e+01f,
                8.618411255e+01f, 8.629707336e+01f, 8.641007233e+01f, 8.652301788e+01f, 8.663594055e+01f, 8.674891663e+01f, 8.686194611e+01f,
                8.604698181e+01f, 8.615980530e+01f, 8.627262115e+01f, 8.638543701e+01f, 8.649829102e+01f, 8.661106873e+01f, 8.672396851e+01f,
                8.590983582e+01f, 8.602252960e+01f, 8.613521576e+01f, 8.624787903e+01f, 8.636055756e+01f, 8.647325134e+01f, 8.658591461e+01f,
                8.577264404e+01f, 8.588522339e+01f, 8.599777985e+01f, 8.611032104e+01f, 8.622286987e+01f, 8.633538818e+01f, 8.644793701e+01f,
                8.563557434e+01f, 8.574797821e+01f, 8.586037445e+01f, 8.597271729e+01f, 8.608514404e+01f, 8.619756317e+01f, 8.630995941e+01f,
                8.549840546e+01f, 8.561072540e+01f, 8.572290802e+01f, 8.583519745e+01f, 8.594744873e+01f, 8.605972290e+01f, 8.617195892e+01f,
                8.536128998e+01f, 8.547340393e+01f, 8.558550262e+01f, 8.569762421e+01f, 8.580971527e+01f, 8.592187500e+01f, 8.603396606e+01f,
                8.522415161e+01f, 8.533609772e+01f, 8.544808960e+01f, 8.556003571e+01f, 8.567201233e+01f, 8.578398132e+01f, 8.589595795e+01f,
                8.508699799e+01f, 8.519882965e+01f, 8.531066895e+01f, 8.542250061e+01f, 8.553430939e+01f, 8.564614868e+01f, 8.575797272e+01f,
                8.494986725e+01f, 8.506156158e+01f, 8.517321014e+01f, 8.528490448e+01f, 8.539659882e+01f, 8.550830078e+01f, 8.561999512e+01f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{inwidth},{inheight}");
        }
    }
}
