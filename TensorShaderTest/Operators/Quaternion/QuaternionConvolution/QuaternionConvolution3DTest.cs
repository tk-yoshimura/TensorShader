using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionConvolution3DTest {
        [TestMethod]
        public void ExecuteFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (4, 4), (4, 8), (8, 4), (8, 20), (20, 32), (32, 8), (32, 36), (36, 36) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                            QuaternionMap3D x = new(inchannels / 4, inwidth, inheight, indepth, batch, xcval);
                            QuaternionFilter3D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, kdepth, wcval);

                            QuaternionMap3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                            QuaternionConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, gradmode: false, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(y_expect, y_actual, 1e-6f, 1e-4f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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

            Random random = new(1234);

            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (4, 4), (4, 8), (8, 4), (8, 20), (20, 32), (32, 8), (32, 36), (36, 36) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {
                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
                            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

                            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                            QuaternionMap3D x = new(inchannels / 4, inwidth, inheight, indepth, batch, xcval);
                            QuaternionFilter3D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, kdepth, wcval);

                            QuaternionMap3D y = Reference(x, w, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, kwidth, kheight, kdepth), wval);

                            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

                            QuaternionConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, gradmode: false, batch);

                            ope.Execute(x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToArray();
                            float[] y_actual = y_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
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

            Random random = new(1234);

            float max_err = 0;

            int batch = 3;
            int inchannels = 196, outchannels = 200;
            int kwidth = 7, kheight = 5, kdepth = 3;
            int inwidth = 125, inheight = 196, indepth = 4;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * inchannels * outchannels / 4]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap3D x = new(inchannels / 4, inwidth, inheight, indepth, batch, xcval);
            QuaternionFilter3D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, kdepth, wcval);

            QuaternionMap3D y = Reference(x, w, kwidth, kheight, kdepth);

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, kwidth, kheight, kdepth), wval);

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch));

            QuaternionConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, gradmode: false, batch);

            ope.Execute(x_tensor, w_tensor, y_tensor);

            float[] y_expect = y.ToArray();
            float[] y_actual = y_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(wval, w_tensor.State.Value);

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedFPTest() {
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.Float;
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;

            int inwidth = 32, inheight = 32, indepth = 32, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            QuaternionConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_convolution_3d_fp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        [TestMethod]
        public void SpeedFFPTest() {
            TensorShaderCudaBackend.Environment.CudnnEnabled = false;
            TensorShaderCudaBackend.Environment.Precision = TensorShaderCudaBackend.Environment.PrecisionMode.FloatFloat;

            int inwidth = 32, inheight = 32, indepth = 32, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor w_tensor = new(Shape.Kernel3D(inchannels, outchannels / 4, ksize, ksize, ksize));

            OverflowCheckedTensor y_tensor = new(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            QuaternionConvolution3D ope = new(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/quaternion_convolution_3d_ffp.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }

        public static QuaternionMap3D Reference(QuaternionMap3D x, QuaternionFilter3D w, int kwidth, int kheight, int kdepth) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw - kwidth + 1, outh = inh - kheight + 1, outd = ind - kdepth + 1;

            QuaternionMap3D y = new(outchannels, outw, outh, outd, batch);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ox, oy, oz = 0; oz < outd; oz++) {
                                for (oy = 0; oy < outh; oy++) {
                                    for (ox = 0; ox < outw; ox++) {
                                        for (int outch = 0; outch < outchannels; outch++) {
                                            Quaternion sum = y[outch, ox, oy, oz, th];

                                            for (int inch = 0; inch < inchannels; inch++) {
                                                sum += x[inch, kx + ox, ky + oy, kz + oz, th] * w[inch, outch, kx, ky, kz];
                                            }

                                            y[outch, ox, oy, oz, th] = sum;
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
            int inchannels = 8, outchannels = 12, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 7, inheight = 8, indepth = 9, batch = 3;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1;

            float[] xval = (new float[batch * inwidth * inheight * indepth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * kdepth * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] wcval = (new Quaternion[wval.Length / 4])
                .Select((_, idx) => new Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

            QuaternionMap3D x = new(inchannels / 4, inwidth, inheight, indepth, batch, xcval);
            QuaternionFilter3D w = new(inchannels / 4, outchannels / 4, kwidth, kheight, kdepth, wcval);

            QuaternionMap3D y = Reference(x, w, kwidth, kheight, kdepth);

            float[] y_expect = {
                -5.067048000e+02f,  5.063402400e+02f,  5.080219200e+02f,  5.062575000e+02f,  -5.017689600e+02f,  5.014144800e+02f,
                5.030894400e+02f,  5.013283800e+02f,  -4.968331200e+02f,  4.964887200e+02f,  4.981569600e+02f,  4.963992600e+02f,
                -5.109585600e+02f,  5.106007200e+02f,  5.122857600e+02f,  5.105146200e+02f,  -5.059958400e+02f,  5.056480800e+02f,
                5.073264000e+02f,  5.055586200e+02f,  -5.010331200e+02f,  5.006954400e+02f,  5.023670400e+02f,  5.006026200e+02f,
                -5.152123200e+02f,  5.148612000e+02f,  5.165496000e+02f,  5.147717400e+02f,  -5.102227200e+02f,  5.098816800e+02f,
                5.115633600e+02f,  5.097888600e+02f,  -5.052331200e+02f,  5.049021600e+02f,  5.065771200e+02f,  5.048059800e+02f,
                -5.194660800e+02f,  5.191216800e+02f,  5.208134400e+02f,  5.190288600e+02f,  -5.144496000e+02f,  5.141152800e+02f,
                5.158003200e+02f,  5.140191000e+02f,  -5.094331200e+02f,  5.091088800e+02f,  5.107872000e+02f,  5.090093400e+02f,
                -5.237198400e+02f,  5.233821600e+02f,  5.250772800e+02f,  5.232859800e+02f,  -5.186764800e+02f,  5.183488800e+02f,
                5.200372800e+02f,  5.182493400e+02f,  -5.136331200e+02f,  5.133156000e+02f,  5.149972800e+02f,  5.132127000e+02f,
                -5.364811200e+02f,  5.361636000e+02f,  5.378688000e+02f,  5.360573400e+02f,  -5.313571200e+02f,  5.310496800e+02f,
                5.327481600e+02f,  5.309400600e+02f,  -5.262331200e+02f,  5.259357600e+02f,  5.276275200e+02f,  5.258227800e+02f,
                -5.407348800e+02f,  5.404240800e+02f,  5.421326400e+02f,  5.403144600e+02f,  -5.355840000e+02f,  5.352832800e+02f,
                5.369851200e+02f,  5.351703000e+02f,  -5.304331200e+02f,  5.301424800e+02f,  5.318376000e+02f,  5.300261400e+02f,
                -5.449886400e+02f,  5.446845600e+02f,  5.463964800e+02f,  5.445715800e+02f,  -5.398108800e+02f,  5.395168800e+02f,
                5.412220800e+02f,  5.394005400e+02f,  -5.346331200e+02f,  5.343492000e+02f,  5.360476800e+02f,  5.342295000e+02f,
                -5.492424000e+02f,  5.489450400e+02f,  5.506603200e+02f,  5.488287000e+02f,  -5.440377600e+02f,  5.437504800e+02f,
                5.454590400e+02f,  5.436307800e+02f,  -5.388331200e+02f,  5.385559200e+02f,  5.402577600e+02f,  5.384328600e+02f,
                -5.534961600e+02f,  5.532055200e+02f,  5.549241600e+02f,  5.530858200e+02f,  -5.482646400e+02f,  5.479840800e+02f,
                5.496960000e+02f,  5.478610200e+02f,  -5.430331200e+02f,  5.427626400e+02f,  5.444678400e+02f,  5.426362200e+02f,
                -5.662574400e+02f,  5.659869600e+02f,  5.677156800e+02f,  5.658571800e+02f,  -5.609452800e+02f,  5.606848800e+02f,
                5.624068800e+02f,  5.605517400e+02f,  -5.556331200e+02f,  5.553828000e+02f,  5.570980800e+02f,  5.552463000e+02f,
                -5.705112000e+02f,  5.702474400e+02f,  5.719795200e+02f,  5.701143000e+02f,  -5.651721600e+02f,  5.649184800e+02f,
                5.666438400e+02f,  5.647819800e+02f,  -5.598331200e+02f,  5.595895200e+02f,  5.613081600e+02f,  5.594496600e+02f,
                -5.747649600e+02f,  5.745079200e+02f,  5.762433600e+02f,  5.743714200e+02f,  -5.693990400e+02f,  5.691520800e+02f,
                5.708808000e+02f,  5.690122200e+02f,  -5.640331200e+02f,  5.637962400e+02f,  5.655182400e+02f,  5.636530200e+02f,
                -5.790187200e+02f,  5.787684000e+02f,  5.805072000e+02f,  5.786285400e+02f,  -5.736259200e+02f,  5.733856800e+02f,
                5.751177600e+02f,  5.732424600e+02f,  -5.682331200e+02f,  5.680029600e+02f,  5.697283200e+02f,  5.678563800e+02f,
                -5.832724800e+02f,  5.830288800e+02f,  5.847710400e+02f,  5.828856600e+02f,  -5.778528000e+02f,  5.776192800e+02f,
                5.793547200e+02f,  5.774727000e+02f,  -5.724331200e+02f,  5.722096800e+02f,  5.739384000e+02f,  5.720597400e+02f,
                -5.960337600e+02f,  5.958103200e+02f,  5.975625600e+02f,  5.956570200e+02f,  -5.905334400e+02f,  5.903200800e+02f,
                5.920656000e+02f,  5.901634200e+02f,  -5.850331200e+02f,  5.848298400e+02f,  5.865686400e+02f,  5.846698200e+02f,
                -6.002875200e+02f,  6.000708000e+02f,  6.018264000e+02f,  5.999141400e+02f,  -5.947603200e+02f,  5.945536800e+02f,
                5.963025600e+02f,  5.943936600e+02f,  -5.892331200e+02f,  5.890365600e+02f,  5.907787200e+02f,  5.888731800e+02f,
                -6.045412800e+02f,  6.043312800e+02f,  6.060902400e+02f,  6.041712600e+02f,  -5.989872000e+02f,  5.987872800e+02f,
                6.005395200e+02f,  5.986239000e+02f,  -5.934331200e+02f,  5.932432800e+02f,  5.949888000e+02f,  5.930765400e+02f,
                -6.087950400e+02f,  6.085917600e+02f,  6.103540800e+02f,  6.084283800e+02f,  -6.032140800e+02f,  6.030208800e+02f,
                6.047764800e+02f,  6.028541400e+02f,  -5.976331200e+02f,  5.974500000e+02f,  5.991988800e+02f,  5.972799000e+02f,
                -6.130488000e+02f,  6.128522400e+02f,  6.146179200e+02f,  6.126855000e+02f,  -6.074409600e+02f,  6.072544800e+02f,
                6.090134400e+02f,  6.070843800e+02f,  -6.018331200e+02f,  6.016567200e+02f,  6.034089600e+02f,  6.014832600e+02f,
                -7.449153600e+02f,  7.449271200e+02f,  7.467969600e+02f,  7.446562200e+02f,  -7.384742400e+02f,  7.384960800e+02f,
                7.403592000e+02f,  7.382218200e+02f,  -7.320331200e+02f,  7.320650400e+02f,  7.339214400e+02f,  7.317874200e+02f,
                -7.491691200e+02f,  7.491876000e+02f,  7.510608000e+02f,  7.489133400e+02f,  -7.427011200e+02f,  7.427296800e+02f,
                7.445961600e+02f,  7.424520600e+02f,  -7.362331200e+02f,  7.362717600e+02f,  7.381315200e+02f,  7.359907800e+02f,
                -7.534228800e+02f,  7.534480800e+02f,  7.553246400e+02f,  7.531704600e+02f,  -7.469280000e+02f,  7.469632800e+02f,
                7.488331200e+02f,  7.466823000e+02f,  -7.404331200e+02f,  7.404784800e+02f,  7.423416000e+02f,  7.401941400e+02f,
                -7.576766400e+02f,  7.577085600e+02f,  7.595884800e+02f,  7.574275800e+02f,  -7.511548800e+02f,  7.511968800e+02f,
                7.530700800e+02f,  7.509125400e+02f,  -7.446331200e+02f,  7.446852000e+02f,  7.465516800e+02f,  7.443975000e+02f,
                -7.619304000e+02f,  7.619690400e+02f,  7.638523200e+02f,  7.616847000e+02f,  -7.553817600e+02f,  7.554304800e+02f,
                7.573070400e+02f,  7.551427800e+02f,  -7.488331200e+02f,  7.488919200e+02f,  7.507617600e+02f,  7.486008600e+02f,
                -7.746916800e+02f,  7.747504800e+02f,  7.766438400e+02f,  7.744560600e+02f,  -7.680624000e+02f,  7.681312800e+02f,
                7.700179200e+02f,  7.678335000e+02f,  -7.614331200e+02f,  7.615120800e+02f,  7.633920000e+02f,  7.612109400e+02f,
                -7.789454400e+02f,  7.790109600e+02f,  7.809076800e+02f,  7.787131800e+02f,  -7.722892800e+02f,  7.723648800e+02f,
                7.742548800e+02f,  7.720637400e+02f,  -7.656331200e+02f,  7.657188000e+02f,  7.676020800e+02f,  7.654143000e+02f,
                -7.831992000e+02f,  7.832714400e+02f,  7.851715200e+02f,  7.829703000e+02f,  -7.765161600e+02f,  7.765984800e+02f,
                7.784918400e+02f,  7.762939800e+02f,  -7.698331200e+02f,  7.699255200e+02f,  7.718121600e+02f,  7.696176600e+02f,
                -7.874529600e+02f,  7.875319200e+02f,  7.894353600e+02f,  7.872274200e+02f,  -7.807430400e+02f,  7.808320800e+02f,
                7.827288000e+02f,  7.805242200e+02f,  -7.740331200e+02f,  7.741322400e+02f,  7.760222400e+02f,  7.738210200e+02f,
                -7.917067200e+02f,  7.917924000e+02f,  7.936992000e+02f,  7.914845400e+02f,  -7.849699200e+02f,  7.850656800e+02f,
                7.869657600e+02f,  7.847544600e+02f,  -7.782331200e+02f,  7.783389600e+02f,  7.802323200e+02f,  7.780243800e+02f,
                -8.044680000e+02f,  8.045738400e+02f,  8.064907200e+02f,  8.042559000e+02f,  -7.976505600e+02f,  7.977664800e+02f,
                7.996766400e+02f,  7.974451800e+02f,  -7.908331200e+02f,  7.909591200e+02f,  7.928625600e+02f,  7.906344600e+02f,
                -8.087217600e+02f,  8.088343200e+02f,  8.107545600e+02f,  8.085130200e+02f,  -8.018774400e+02f,  8.020000800e+02f,
                8.039136000e+02f,  8.016754200e+02f,  -7.950331200e+02f,  7.951658400e+02f,  7.970726400e+02f,  7.948378200e+02f,
                -8.129755200e+02f,  8.130948000e+02f,  8.150184000e+02f,  8.127701400e+02f,  -8.061043200e+02f,  8.062336800e+02f,
                8.081505600e+02f,  8.059056600e+02f,  -7.992331200e+02f,  7.993725600e+02f,  8.012827200e+02f,  7.990411800e+02f,
                -8.172292800e+02f,  8.173552800e+02f,  8.192822400e+02f,  8.170272600e+02f,  -8.103312000e+02f,  8.104672800e+02f,
                8.123875200e+02f,  8.101359000e+02f,  -8.034331200e+02f,  8.035792800e+02f,  8.054928000e+02f,  8.032445400e+02f,
                -8.214830400e+02f,  8.216157600e+02f,  8.235460800e+02f,  8.212843800e+02f,  -8.145580800e+02f,  8.147008800e+02f,
                8.166244800e+02f,  8.143661400e+02f,  -8.076331200e+02f,  8.077860000e+02f,  8.097028800e+02f,  8.074479000e+02f,
                -8.342443200e+02f,  8.343972000e+02f,  8.363376000e+02f,  8.340557400e+02f,  -8.272387200e+02f,  8.274016800e+02f,
                8.293353600e+02f,  8.270568600e+02f,  -8.202331200e+02f,  8.204061600e+02f,  8.223331200e+02f,  8.200579800e+02f,
                -8.384980800e+02f,  8.386576800e+02f,  8.406014400e+02f,  8.383128600e+02f,  -8.314656000e+02f,  8.316352800e+02f,
                8.335723200e+02f,  8.312871000e+02f,  -8.244331200e+02f,  8.246128800e+02f,  8.265432000e+02f,  8.242613400e+02f,
                -8.427518400e+02f,  8.429181600e+02f,  8.448652800e+02f,  8.425699800e+02f,  -8.356924800e+02f,  8.358688800e+02f,
                8.378092800e+02f,  8.355173400e+02f,  -8.286331200e+02f,  8.288196000e+02f,  8.307532800e+02f,  8.284647000e+02f,
                -8.470056000e+02f,  8.471786400e+02f,  8.491291200e+02f,  8.468271000e+02f,  -8.399193600e+02f,  8.401024800e+02f,
                8.420462400e+02f,  8.397475800e+02f,  -8.328331200e+02f,  8.330263200e+02f,  8.349633600e+02f,  8.326680600e+02f,
                -8.512593600e+02f,  8.514391200e+02f,  8.533929600e+02f,  8.510842200e+02f,  -8.441462400e+02f,  8.443360800e+02f,
                8.462832000e+02f,  8.439778200e+02f,  -8.370331200e+02f,  8.372330400e+02f,  8.391734400e+02f,  8.368714200e+02f,
                -9.831259200e+02f,  9.835140000e+02f,  9.855720000e+02f,  9.830549400e+02f,  -9.751795200e+02f,  9.755776800e+02f,
                9.776289600e+02f,  9.751152600e+02f,  -9.672331200e+02f,  9.676413600e+02f,  9.696859200e+02f,  9.671755800e+02f,
                -9.873796800e+02f,  9.877744800e+02f,  9.898358400e+02f,  9.873120600e+02f,  -9.794064000e+02f,  9.798112800e+02f,
                9.818659200e+02f,  9.793455000e+02f,  -9.714331200e+02f,  9.718480800e+02f,  9.738960000e+02f,  9.713789400e+02f,
                -9.916334400e+02f,  9.920349600e+02f,  9.940996800e+02f,  9.915691800e+02f,  -9.836332800e+02f,  9.840448800e+02f,
                9.861028800e+02f,  9.835757400e+02f,  -9.756331200e+02f,  9.760548000e+02f,  9.781060800e+02f,  9.755823000e+02f,
                -9.958872000e+02f,  9.962954400e+02f,  9.983635200e+02f,  9.958263000e+02f,  -9.878601600e+02f,  9.882784800e+02f,
                9.903398400e+02f,  9.878059800e+02f,  -9.798331200e+02f,  9.802615200e+02f,  9.823161600e+02f,  9.797856600e+02f,
                -1.000140960e+03f,  1.000555920e+03f,  1.002627360e+03f,  1.000083420e+03f,  -9.920870400e+02f,  9.925120800e+02f,
                9.945768000e+02f,  9.920362200e+02f,  -9.840331200e+02f,  9.844682400e+02f,  9.865262400e+02f,  9.839890200e+02f,
                -1.012902240e+03f,  1.013337360e+03f,  1.015418880e+03f,  1.012854780e+03f,  -1.004767680e+03f,  1.005212880e+03f,
                1.007287680e+03f,  1.004726940e+03f,  -9.966331200e+02f,  9.970884000e+02f,  9.991564800e+02f,  9.965991000e+02f,
                -1.017156000e+03f,  1.017597840e+03f,  1.019682720e+03f,  1.017111900e+03f,  -1.008994560e+03f,  1.009446480e+03f,
                1.011524640e+03f,  1.008957180e+03f,  -1.000833120e+03f,  1.001295120e+03f,  1.003366560e+03f,  1.000802460e+03f,
                -1.021409760e+03f,  1.021858320e+03f,  1.023946560e+03f,  1.021369020e+03f,  -1.013221440e+03f,  1.013680080e+03f,
                1.015761600e+03f,  1.013187420e+03f,  -1.005033120e+03f,  1.005501840e+03f,  1.007576640e+03f,  1.005005820e+03f,
                -1.025663520e+03f,  1.026118800e+03f,  1.028210400e+03f,  1.025626140e+03f,  -1.017448320e+03f,  1.017913680e+03f,
                1.019998560e+03f,  1.017417660e+03f,  -1.009233120e+03f,  1.009708560e+03f,  1.011786720e+03f,  1.009209180e+03f,
                -1.029917280e+03f,  1.030379280e+03f,  1.032474240e+03f,  1.029883260e+03f,  -1.021675200e+03f,  1.022147280e+03f,
                1.024235520e+03f,  1.021647900e+03f,  -1.013433120e+03f,  1.013915280e+03f,  1.015996800e+03f,  1.013412540e+03f,
                -1.042678560e+03f,  1.043160720e+03f,  1.045265760e+03f,  1.042654620e+03f,  -1.034355840e+03f,  1.034848080e+03f,
                1.036946400e+03f,  1.034338620e+03f,  -1.026033120e+03f,  1.026535440e+03f,  1.028627040e+03f,  1.026022620e+03f,
                -1.046932320e+03f,  1.047421200e+03f,  1.049529600e+03f,  1.046911740e+03f,  -1.038582720e+03f,  1.039081680e+03f,
                1.041183360e+03f,  1.038568860e+03f,  -1.030233120e+03f,  1.030742160e+03f,  1.032837120e+03f,  1.030225980e+03f,
                -1.051186080e+03f,  1.051681680e+03f,  1.053793440e+03f,  1.051168860e+03f,  -1.042809600e+03f,  1.043315280e+03f,
                1.045420320e+03f,  1.042799100e+03f,  -1.034433120e+03f,  1.034948880e+03f,  1.037047200e+03f,  1.034429340e+03f,
                -1.055439840e+03f,  1.055942160e+03f,  1.058057280e+03f,  1.055425980e+03f,  -1.047036480e+03f,  1.047548880e+03f,
                1.049657280e+03f,  1.047029340e+03f,  -1.038633120e+03f,  1.039155600e+03f,  1.041257280e+03f,  1.038632700e+03f,
                -1.059693600e+03f,  1.060202640e+03f,  1.062321120e+03f,  1.059683100e+03f,  -1.051263360e+03f,  1.051782480e+03f,
                1.053894240e+03f,  1.051259580e+03f,  -1.042833120e+03f,  1.043362320e+03f,  1.045467360e+03f,  1.042836060e+03f,
                -1.072454880e+03f,  1.072984080e+03f,  1.075112640e+03f,  1.072454460e+03f,  -1.063944000e+03f,  1.064483280e+03f,
                1.066605120e+03f,  1.063950300e+03f,  -1.055433120e+03f,  1.055982480e+03f,  1.058097600e+03f,  1.055446140e+03f,
                -1.076708640e+03f,  1.077244560e+03f,  1.079376480e+03f,  1.076711580e+03f,  -1.068170880e+03f,  1.068716880e+03f,
                1.070842080e+03f,  1.068180540e+03f,  -1.059633120e+03f,  1.060189200e+03f,  1.062307680e+03f,  1.059649500e+03f,
                -1.080962400e+03f,  1.081505040e+03f,  1.083640320e+03f,  1.080968700e+03f,  -1.072397760e+03f,  1.072950480e+03f,
                1.075079040e+03f,  1.072410780e+03f,  -1.063833120e+03f,  1.064395920e+03f,  1.066517760e+03f,  1.063852860e+03f,
                -1.085216160e+03f,  1.085765520e+03f,  1.087904160e+03f,  1.085225820e+03f,  -1.076624640e+03f,  1.077184080e+03f,
                1.079316000e+03f,  1.076641020e+03f,  -1.068033120e+03f,  1.068602640e+03f,  1.070727840e+03f,  1.068056220e+03f,
                -1.089469920e+03f,  1.090026000e+03f,  1.092168000e+03f,  1.089482940e+03f,  -1.080851520e+03f,  1.081417680e+03f,
                1.083552960e+03f,  1.080871260e+03f,  -1.072233120e+03f,  1.072809360e+03f,  1.074937920e+03f,  1.072259580e+03f,
                -2.650599840e+03f,  2.653622160e+03f,  2.656997280e+03f,  2.651845980e+03f,  -2.632116480e+03f,  2.635148880e+03f,
                2.638517280e+03f,  2.633369340e+03f,  -2.613633120e+03f,  2.616675600e+03f,  2.620037280e+03f,  2.614892700e+03f,
                -2.654853600e+03f,  2.657882640e+03f,  2.661261120e+03f,  2.656103100e+03f,  -2.636343360e+03f,  2.639382480e+03f,
                2.642754240e+03f,  2.637599580e+03f,  -2.617833120e+03f,  2.620882320e+03f,  2.624247360e+03f,  2.619096060e+03f,
                -2.659107360e+03f,  2.662143120e+03f,  2.665524960e+03f,  2.660360220e+03f,  -2.640570240e+03f,  2.643616080e+03f,
                2.646991200e+03f,  2.641829820e+03f,  -2.622033120e+03f,  2.625089040e+03f,  2.628457440e+03f,  2.623299420e+03f,
                -2.663361120e+03f,  2.666403600e+03f,  2.669788800e+03f,  2.664617340e+03f,  -2.644797120e+03f,  2.647849680e+03f,
                2.651228160e+03f,  2.646060060e+03f,  -2.626233120e+03f,  2.629295760e+03f,  2.632667520e+03f,  2.627502780e+03f,
                -2.667614880e+03f,  2.670664080e+03f,  2.674052640e+03f,  2.668874460e+03f,  -2.649024000e+03f,  2.652083280e+03f,
                2.655465120e+03f,  2.650290300e+03f,  -2.630433120e+03f,  2.633502480e+03f,  2.636877600e+03f,  2.631706140e+03f,
                -2.680376160e+03f,  2.683445520e+03f,  2.686844160e+03f,  2.681645820e+03f,  -2.661704640e+03f,  2.664784080e+03f,
                2.668176000e+03f,  2.662981020e+03f,  -2.643033120e+03f,  2.646122640e+03f,  2.649507840e+03f,  2.644316220e+03f,
                -2.684629920e+03f,  2.687706000e+03f,  2.691108000e+03f,  2.685902940e+03f,  -2.665931520e+03f,  2.669017680e+03f,
                2.672412960e+03f,  2.667211260e+03f,  -2.647233120e+03f,  2.650329360e+03f,  2.653717920e+03f,  2.648519580e+03f,
                -2.688883680e+03f,  2.691966480e+03f,  2.695371840e+03f,  2.690160060e+03f,  -2.670158400e+03f,  2.673251280e+03f,
                2.676649920e+03f,  2.671441500e+03f,  -2.651433120e+03f,  2.654536080e+03f,  2.657928000e+03f,  2.652722940e+03f,
                -2.693137440e+03f,  2.696226960e+03f,  2.699635680e+03f,  2.694417180e+03f,  -2.674385280e+03f,  2.677484880e+03f,
                2.680886880e+03f,  2.675671740e+03f,  -2.655633120e+03f,  2.658742800e+03f,  2.662138080e+03f,  2.656926300e+03f,
                -2.697391200e+03f,  2.700487440e+03f,  2.703899520e+03f,  2.698674300e+03f,  -2.678612160e+03f,  2.681718480e+03f,
                2.685123840e+03f,  2.679901980e+03f,  -2.659833120e+03f,  2.662949520e+03f,  2.666348160e+03f,  2.661129660e+03f,
                -2.710152480e+03f,  2.713268880e+03f,  2.716691040e+03f,  2.711445660e+03f,  -2.691292800e+03f,  2.694419280e+03f,
                2.697834720e+03f,  2.692592700e+03f,  -2.672433120e+03f,  2.675569680e+03f,  2.678978400e+03f,  2.673739740e+03f,
                -2.714406240e+03f,  2.717529360e+03f,  2.720954880e+03f,  2.715702780e+03f,  -2.695519680e+03f,  2.698652880e+03f,
                2.702071680e+03f,  2.696822940e+03f,  -2.676633120e+03f,  2.679776400e+03f,  2.683188480e+03f,  2.677943100e+03f,
                -2.718660000e+03f,  2.721789840e+03f,  2.725218720e+03f,  2.719959900e+03f,  -2.699746560e+03f,  2.702886480e+03f,
                2.706308640e+03f,  2.701053180e+03f,  -2.680833120e+03f,  2.683983120e+03f,  2.687398560e+03f,  2.682146460e+03f,
                -2.722913760e+03f,  2.726050320e+03f,  2.729482560e+03f,  2.724217020e+03f,  -2.703973440e+03f,  2.707120080e+03f,
                2.710545600e+03f,  2.705283420e+03f,  -2.685033120e+03f,  2.688189840e+03f,  2.691608640e+03f,  2.686349820e+03f,
                -2.727167520e+03f,  2.730310800e+03f,  2.733746400e+03f,  2.728474140e+03f,  -2.708200320e+03f,  2.711353680e+03f,
                2.714782560e+03f,  2.709513660e+03f,  -2.689233120e+03f,  2.692396560e+03f,  2.695818720e+03f,  2.690553180e+03f,
                -2.739928800e+03f,  2.743092240e+03f,  2.746537920e+03f,  2.741245500e+03f,  -2.720880960e+03f,  2.724054480e+03f,
                2.727493440e+03f,  2.722204380e+03f,  -2.701833120e+03f,  2.705016720e+03f,  2.708448960e+03f,  2.703163260e+03f,
                -2.744182560e+03f,  2.747352720e+03f,  2.750801760e+03f,  2.745502620e+03f,  -2.725107840e+03f,  2.728288080e+03f,
                2.731730400e+03f,  2.726434620e+03f,  -2.706033120e+03f,  2.709223440e+03f,  2.712659040e+03f,  2.707366620e+03f,
                -2.748436320e+03f,  2.751613200e+03f,  2.755065600e+03f,  2.749759740e+03f,  -2.729334720e+03f,  2.732521680e+03f,
                2.735967360e+03f,  2.730664860e+03f,  -2.710233120e+03f,  2.713430160e+03f,  2.716869120e+03f,  2.711569980e+03f,
                -2.752690080e+03f,  2.755873680e+03f,  2.759329440e+03f,  2.754016860e+03f,  -2.733561600e+03f,  2.736755280e+03f,
                2.740204320e+03f,  2.734895100e+03f,  -2.714433120e+03f,  2.717636880e+03f,  2.721079200e+03f,  2.715773340e+03f,
                -2.756943840e+03f,  2.760134160e+03f,  2.763593280e+03f,  2.758273980e+03f,  -2.737788480e+03f,  2.740988880e+03f,
                2.744441280e+03f,  2.739125340e+03f,  -2.718633120e+03f,  2.721843600e+03f,  2.725289280e+03f,  2.719976700e+03f,
                -2.888810400e+03f,  2.892209040e+03f,  2.895772320e+03f,  2.890244700e+03f,  -2.868821760e+03f,  2.872230480e+03f,
                2.875787040e+03f,  2.870262780e+03f,  -2.848833120e+03f,  2.852251920e+03f,  2.855801760e+03f,  2.850280860e+03f,
                -2.893064160e+03f,  2.896469520e+03f,  2.900036160e+03f,  2.894501820e+03f,  -2.873048640e+03f,  2.876464080e+03f,
                2.880024000e+03f,  2.874493020e+03f,  -2.853033120e+03f,  2.856458640e+03f,  2.860011840e+03f,  2.854484220e+03f,
                -2.897317920e+03f,  2.900730000e+03f,  2.904300000e+03f,  2.898758940e+03f,  -2.877275520e+03f,  2.880697680e+03f,
                2.884260960e+03f,  2.878723260e+03f,  -2.857233120e+03f,  2.860665360e+03f,  2.864221920e+03f,  2.858687580e+03f,
                -2.901571680e+03f,  2.904990480e+03f,  2.908563840e+03f,  2.903016060e+03f,  -2.881502400e+03f,  2.884931280e+03f,
                2.888497920e+03f,  2.882953500e+03f,  -2.861433120e+03f,  2.864872080e+03f,  2.868432000e+03f,  2.862890940e+03f,
                -2.905825440e+03f,  2.909250960e+03f,  2.912827680e+03f,  2.907273180e+03f,  -2.885729280e+03f,  2.889164880e+03f,
                2.892734880e+03f,  2.887183740e+03f,  -2.865633120e+03f,  2.869078800e+03f,  2.872642080e+03f,  2.867094300e+03f,
                -2.918586720e+03f,  2.922032400e+03f,  2.925619200e+03f,  2.920044540e+03f,  -2.898409920e+03f,  2.901865680e+03f,
                2.905445760e+03f,  2.899874460e+03f,  -2.878233120e+03f,  2.881698960e+03f,  2.885272320e+03f,  2.879704380e+03f,
                -2.922840480e+03f,  2.926292880e+03f,  2.929883040e+03f,  2.924301660e+03f,  -2.902636800e+03f,  2.906099280e+03f,
                2.909682720e+03f,  2.904104700e+03f,  -2.882433120e+03f,  2.885905680e+03f,  2.889482400e+03f,  2.883907740e+03f,
                -2.927094240e+03f,  2.930553360e+03f,  2.934146880e+03f,  2.928558780e+03f,  -2.906863680e+03f,  2.910332880e+03f,
                2.913919680e+03f,  2.908334940e+03f,  -2.886633120e+03f,  2.890112400e+03f,  2.893692480e+03f,  2.888111100e+03f,
                -2.931348000e+03f,  2.934813840e+03f,  2.938410720e+03f,  2.932815900e+03f,  -2.911090560e+03f,  2.914566480e+03f,
                2.918156640e+03f,  2.912565180e+03f,  -2.890833120e+03f,  2.894319120e+03f,  2.897902560e+03f,  2.892314460e+03f,
                -2.935601760e+03f,  2.939074320e+03f,  2.942674560e+03f,  2.937073020e+03f,  -2.915317440e+03f,  2.918800080e+03f,
                2.922393600e+03f,  2.916795420e+03f,  -2.895033120e+03f,  2.898525840e+03f,  2.902112640e+03f,  2.896517820e+03f,
                -2.948363040e+03f,  2.951855760e+03f,  2.955466080e+03f,  2.949844380e+03f,  -2.927998080e+03f,  2.931500880e+03f,
                2.935104480e+03f,  2.929486140e+03f,  -2.907633120e+03f,  2.911146000e+03f,  2.914742880e+03f,  2.909127900e+03f,
                -2.952616800e+03f,  2.956116240e+03f,  2.959729920e+03f,  2.954101500e+03f,  -2.932224960e+03f,  2.935734480e+03f,
                2.939341440e+03f,  2.933716380e+03f,  -2.911833120e+03f,  2.915352720e+03f,  2.918952960e+03f,  2.913331260e+03f,
                -2.956870560e+03f,  2.960376720e+03f,  2.963993760e+03f,  2.958358620e+03f,  -2.936451840e+03f,  2.939968080e+03f,
                2.943578400e+03f,  2.937946620e+03f,  -2.916033120e+03f,  2.919559440e+03f,  2.923163040e+03f,  2.917534620e+03f,
                -2.961124320e+03f,  2.964637200e+03f,  2.968257600e+03f,  2.962615740e+03f,  -2.940678720e+03f,  2.944201680e+03f,
                2.947815360e+03f,  2.942176860e+03f,  -2.920233120e+03f,  2.923766160e+03f,  2.927373120e+03f,  2.921737980e+03f,
                -2.965378080e+03f,  2.968897680e+03f,  2.972521440e+03f,  2.966872860e+03f,  -2.944905600e+03f,  2.948435280e+03f,
                2.952052320e+03f,  2.946407100e+03f,  -2.924433120e+03f,  2.927972880e+03f,  2.931583200e+03f,  2.925941340e+03f,
                -2.978139360e+03f,  2.981679120e+03f,  2.985312960e+03f,  2.979644220e+03f,  -2.957586240e+03f,  2.961136080e+03f,
                2.964763200e+03f,  2.959097820e+03f,  -2.937033120e+03f,  2.940593040e+03f,  2.944213440e+03f,  2.938551420e+03f,
                -2.982393120e+03f,  2.985939600e+03f,  2.989576800e+03f,  2.983901340e+03f,  -2.961813120e+03f,  2.965369680e+03f,
                2.969000160e+03f,  2.963328060e+03f,  -2.941233120e+03f,  2.944799760e+03f,  2.948423520e+03f,  2.942754780e+03f,
                -2.986646880e+03f,  2.990200080e+03f,  2.993840640e+03f,  2.988158460e+03f,  -2.966040000e+03f,  2.969603280e+03f,
                2.973237120e+03f,  2.967558300e+03f,  -2.945433120e+03f,  2.949006480e+03f,  2.952633600e+03f,  2.946958140e+03f,
                -2.990900640e+03f,  2.994460560e+03f,  2.998104480e+03f,  2.992415580e+03f,  -2.970266880e+03f,  2.973836880e+03f,
                2.977474080e+03f,  2.971788540e+03f,  -2.949633120e+03f,  2.953213200e+03f,  2.956843680e+03f,  2.951161500e+03f,
                -2.995154400e+03f,  2.998721040e+03f,  3.002368320e+03f,  2.996672700e+03f,  -2.974493760e+03f,  2.978070480e+03f,
                2.981711040e+03f,  2.976018780e+03f,  -2.953833120e+03f,  2.957419920e+03f,  2.961053760e+03f,  2.955364860e+03f,
                -3.127020960e+03f,  3.130795920e+03f,  3.134547360e+03f,  3.128643420e+03f,  -3.105527040e+03f,  3.109312080e+03f,
                3.113056800e+03f,  3.107156220e+03f,  -3.084033120e+03f,  3.087828240e+03f,  3.091566240e+03f,  3.085669020e+03f,
                -3.131274720e+03f,  3.135056400e+03f,  3.138811200e+03f,  3.132900540e+03f,  -3.109753920e+03f,  3.113545680e+03f,
                3.117293760e+03f,  3.111386460e+03f,  -3.088233120e+03f,  3.092034960e+03f,  3.095776320e+03f,  3.089872380e+03f,
                -3.135528480e+03f,  3.139316880e+03f,  3.143075040e+03f,  3.137157660e+03f,  -3.113980800e+03f,  3.117779280e+03f,
                3.121530720e+03f,  3.115616700e+03f,  -3.092433120e+03f,  3.096241680e+03f,  3.099986400e+03f,  3.094075740e+03f,
                -3.139782240e+03f,  3.143577360e+03f,  3.147338880e+03f,  3.141414780e+03f,  -3.118207680e+03f,  3.122012880e+03f,
                3.125767680e+03f,  3.119846940e+03f,  -3.096633120e+03f,  3.100448400e+03f,  3.104196480e+03f,  3.098279100e+03f,
                -3.144036000e+03f,  3.147837840e+03f,  3.151602720e+03f,  3.145671900e+03f,  -3.122434560e+03f,  3.126246480e+03f,
                3.130004640e+03f,  3.124077180e+03f,  -3.100833120e+03f,  3.104655120e+03f,  3.108406560e+03f,  3.102482460e+03f,
                -3.156797280e+03f,  3.160619280e+03f,  3.164394240e+03f,  3.158443260e+03f,  -3.135115200e+03f,  3.138947280e+03f,
                3.142715520e+03f,  3.136767900e+03f,  -3.113433120e+03f,  3.117275280e+03f,  3.121036800e+03f,  3.115092540e+03f,
                -3.161051040e+03f,  3.164879760e+03f,  3.168658080e+03f,  3.162700380e+03f,  -3.139342080e+03f,  3.143180880e+03f,
                3.146952480e+03f,  3.140998140e+03f,  -3.117633120e+03f,  3.121482000e+03f,  3.125246880e+03f,  3.119295900e+03f,
                -3.165304800e+03f,  3.169140240e+03f,  3.172921920e+03f,  3.166957500e+03f,  -3.143568960e+03f,  3.147414480e+03f,
                3.151189440e+03f,  3.145228380e+03f,  -3.121833120e+03f,  3.125688720e+03f,  3.129456960e+03f,  3.123499260e+03f,
                -3.169558560e+03f,  3.173400720e+03f,  3.177185760e+03f,  3.171214620e+03f,  -3.147795840e+03f,  3.151648080e+03f,
                3.155426400e+03f,  3.149458620e+03f,  -3.126033120e+03f,  3.129895440e+03f,  3.133667040e+03f,  3.127702620e+03f,
                -3.173812320e+03f,  3.177661200e+03f,  3.181449600e+03f,  3.175471740e+03f,  -3.152022720e+03f,  3.155881680e+03f,
                3.159663360e+03f,  3.153688860e+03f,  -3.130233120e+03f,  3.134102160e+03f,  3.137877120e+03f,  3.131905980e+03f,
                -3.186573600e+03f,  3.190442640e+03f,  3.194241120e+03f,  3.188243100e+03f,  -3.164703360e+03f,  3.168582480e+03f,
                3.172374240e+03f,  3.166379580e+03f,  -3.142833120e+03f,  3.146722320e+03f,  3.150507360e+03f,  3.144516060e+03f,
                -3.190827360e+03f,  3.194703120e+03f,  3.198504960e+03f,  3.192500220e+03f,  -3.168930240e+03f,  3.172816080e+03f,
                3.176611200e+03f,  3.170609820e+03f,  -3.147033120e+03f,  3.150929040e+03f,  3.154717440e+03f,  3.148719420e+03f,
                -3.195081120e+03f,  3.198963600e+03f,  3.202768800e+03f,  3.196757340e+03f,  -3.173157120e+03f,  3.177049680e+03f,
                3.180848160e+03f,  3.174840060e+03f,  -3.151233120e+03f,  3.155135760e+03f,  3.158927520e+03f,  3.152922780e+03f,
                -3.199334880e+03f,  3.203224080e+03f,  3.207032640e+03f,  3.201014460e+03f,  -3.177384000e+03f,  3.181283280e+03f,
                3.185085120e+03f,  3.179070300e+03f,  -3.155433120e+03f,  3.159342480e+03f,  3.163137600e+03f,  3.157126140e+03f,
                -3.203588640e+03f,  3.207484560e+03f,  3.211296480e+03f,  3.205271580e+03f,  -3.181610880e+03f,  3.185516880e+03f,
                3.189322080e+03f,  3.183300540e+03f,  -3.159633120e+03f,  3.163549200e+03f,  3.167347680e+03f,  3.161329500e+03f,
                -3.216349920e+03f,  3.220266000e+03f,  3.224088000e+03f,  3.218042940e+03f,  -3.194291520e+03f,  3.198217680e+03f,
                3.202032960e+03f,  3.195991260e+03f,  -3.172233120e+03f,  3.176169360e+03f,  3.179977920e+03f,  3.173939580e+03f,
                -3.220603680e+03f,  3.224526480e+03f,  3.228351840e+03f,  3.222300060e+03f,  -3.198518400e+03f,  3.202451280e+03f,
                3.206269920e+03f,  3.200221500e+03f,  -3.176433120e+03f,  3.180376080e+03f,  3.184188000e+03f,  3.178142940e+03f,
                -3.224857440e+03f,  3.228786960e+03f,  3.232615680e+03f,  3.226557180e+03f,  -3.202745280e+03f,  3.206684880e+03f,
                3.210506880e+03f,  3.204451740e+03f,  -3.180633120e+03f,  3.184582800e+03f,  3.188398080e+03f,  3.182346300e+03f,
                -3.229111200e+03f,  3.233047440e+03f,  3.236879520e+03f,  3.230814300e+03f,  -3.206972160e+03f,  3.210918480e+03f,
                3.214743840e+03f,  3.208681980e+03f,  -3.184833120e+03f,  3.188789520e+03f,  3.192608160e+03f,  3.186549660e+03f,
                -3.233364960e+03f,  3.237307920e+03f,  3.241143360e+03f,  3.235071420e+03f,  -3.211199040e+03f,  3.215152080e+03f,
                3.218980800e+03f,  3.212912220e+03f,  -3.189033120e+03f,  3.192996240e+03f,  3.196818240e+03f,  3.190753020e+03f,
                -4.794494880e+03f,  4.800904080e+03f,  4.805972640e+03f,  4.797434460e+03f,  -4.762464000e+03f,  4.768883280e+03f,
                4.773945120e+03f,  4.765410300e+03f,  -4.730433120e+03f,  4.736862480e+03f,  4.741917600e+03f,  4.733386140e+03f,
                -4.798748640e+03f,  4.805164560e+03f,  4.810236480e+03f,  4.801691580e+03f,  -4.766690880e+03f,  4.773116880e+03f,
                4.778182080e+03f,  4.769640540e+03f,  -4.734633120e+03f,  4.741069200e+03f,  4.746127680e+03f,  4.737589500e+03f,
                -4.803002400e+03f,  4.809425040e+03f,  4.814500320e+03f,  4.805948700e+03f,  -4.770917760e+03f,  4.777350480e+03f,
                4.782419040e+03f,  4.773870780e+03f,  -4.738833120e+03f,  4.745275920e+03f,  4.750337760e+03f,  4.741792860e+03f,
                -4.807256160e+03f,  4.813685520e+03f,  4.818764160e+03f,  4.810205820e+03f,  -4.775144640e+03f,  4.781584080e+03f,
                4.786656000e+03f,  4.778101020e+03f,  -4.743033120e+03f,  4.749482640e+03f,  4.754547840e+03f,  4.745996220e+03f,
                -4.811509920e+03f,  4.817946000e+03f,  4.823028000e+03f,  4.814462940e+03f,  -4.779371520e+03f,  4.785817680e+03f,
                4.790892960e+03f,  4.782331260e+03f,  -4.747233120e+03f,  4.753689360e+03f,  4.758757920e+03f,  4.750199580e+03f,
                -4.824271200e+03f,  4.830727440e+03f,  4.835819520e+03f,  4.827234300e+03f,  -4.792052160e+03f,  4.798518480e+03f,
                4.803603840e+03f,  4.795021980e+03f,  -4.759833120e+03f,  4.766309520e+03f,  4.771388160e+03f,  4.762809660e+03f,
                -4.828524960e+03f,  4.834987920e+03f,  4.840083360e+03f,  4.831491420e+03f,  -4.796279040e+03f,  4.802752080e+03f,
                4.807840800e+03f,  4.799252220e+03f,  -4.764033120e+03f,  4.770516240e+03f,  4.775598240e+03f,  4.767013020e+03f,
                -4.832778720e+03f,  4.839248400e+03f,  4.844347200e+03f,  4.835748540e+03f,  -4.800505920e+03f,  4.806985680e+03f,
                4.812077760e+03f,  4.803482460e+03f,  -4.768233120e+03f,  4.774722960e+03f,  4.779808320e+03f,  4.771216380e+03f,
                -4.837032480e+03f,  4.843508880e+03f,  4.848611040e+03f,  4.840005660e+03f,  -4.804732800e+03f,  4.811219280e+03f,
                4.816314720e+03f,  4.807712700e+03f,  -4.772433120e+03f,  4.778929680e+03f,  4.784018400e+03f,  4.775419740e+03f,
                -4.841286240e+03f,  4.847769360e+03f,  4.852874880e+03f,  4.844262780e+03f,  -4.808959680e+03f,  4.815452880e+03f,
                4.820551680e+03f,  4.811942940e+03f,  -4.776633120e+03f,  4.783136400e+03f,  4.788228480e+03f,  4.779623100e+03f,
                -4.854047520e+03f,  4.860550800e+03f,  4.865666400e+03f,  4.857034140e+03f,  -4.821640320e+03f,  4.828153680e+03f,
                4.833262560e+03f,  4.824633660e+03f,  -4.789233120e+03f,  4.795756560e+03f,  4.800858720e+03f,  4.792233180e+03f,
                -4.858301280e+03f,  4.864811280e+03f,  4.869930240e+03f,  4.861291260e+03f,  -4.825867200e+03f,  4.832387280e+03f,
                4.837499520e+03f,  4.828863900e+03f,  -4.793433120e+03f,  4.799963280e+03f,  4.805068800e+03f,  4.796436540e+03f,
                -4.862555040e+03f,  4.869071760e+03f,  4.874194080e+03f,  4.865548380e+03f,  -4.830094080e+03f,  4.836620880e+03f,
                4.841736480e+03f,  4.833094140e+03f,  -4.797633120e+03f,  4.804170000e+03f,  4.809278880e+03f,  4.800639900e+03f,
                -4.866808800e+03f,  4.873332240e+03f,  4.878457920e+03f,  4.869805500e+03f,  -4.834320960e+03f,  4.840854480e+03f,
                4.845973440e+03f,  4.837324380e+03f,  -4.801833120e+03f,  4.808376720e+03f,  4.813488960e+03f,  4.804843260e+03f,
                -4.871062560e+03f,  4.877592720e+03f,  4.882721760e+03f,  4.874062620e+03f,  -4.838547840e+03f,  4.845088080e+03f,
                4.850210400e+03f,  4.841554620e+03f,  -4.806033120e+03f,  4.812583440e+03f,  4.817699040e+03f,  4.809046620e+03f,
                -4.883823840e+03f,  4.890374160e+03f,  4.895513280e+03f,  4.886833980e+03f,  -4.851228480e+03f,  4.857788880e+03f,
                4.862921280e+03f,  4.854245340e+03f,  -4.818633120e+03f,  4.825203600e+03f,  4.830329280e+03f,  4.821656700e+03f,
                -4.888077600e+03f,  4.894634640e+03f,  4.899777120e+03f,  4.891091100e+03f,  -4.855455360e+03f,  4.862022480e+03f,
                4.867158240e+03f,  4.858475580e+03f,  -4.822833120e+03f,  4.829410320e+03f,  4.834539360e+03f,  4.825860060e+03f,
                -4.892331360e+03f,  4.898895120e+03f,  4.904040960e+03f,  4.895348220e+03f,  -4.859682240e+03f,  4.866256080e+03f,
                4.871395200e+03f,  4.862705820e+03f,  -4.827033120e+03f,  4.833617040e+03f,  4.838749440e+03f,  4.830063420e+03f,
                -4.896585120e+03f,  4.903155600e+03f,  4.908304800e+03f,  4.899605340e+03f,  -4.863909120e+03f,  4.870489680e+03f,
                4.875632160e+03f,  4.866936060e+03f,  -4.831233120e+03f,  4.837823760e+03f,  4.842959520e+03f,  4.834266780e+03f,
                -4.900838880e+03f,  4.907416080e+03f,  4.912568640e+03f,  4.903862460e+03f,  -4.868136000e+03f,  4.874723280e+03f,
                4.879869120e+03f,  4.871166300e+03f,  -4.835433120e+03f,  4.842030480e+03f,  4.847169600e+03f,  4.838470140e+03f,
                -5.032705440e+03f,  5.039490960e+03f,  5.044747680e+03f,  5.035833180e+03f,  -4.999169280e+03f,  5.005964880e+03f,
                5.011214880e+03f,  5.002303740e+03f,  -4.965633120e+03f,  4.972438800e+03f,  4.977682080e+03f,  4.968774300e+03f,
                -5.036959200e+03f,  5.043751440e+03f,  5.049011520e+03f,  5.040090300e+03f,  -5.003396160e+03f,  5.010198480e+03f,
                5.015451840e+03f,  5.006533980e+03f,  -4.969833120e+03f,  4.976645520e+03f,  4.981892160e+03f,  4.972977660e+03f,
                -5.041212960e+03f,  5.048011920e+03f,  5.053275360e+03f,  5.044347420e+03f,  -5.007623040e+03f,  5.014432080e+03f,
                5.019688800e+03f,  5.010764220e+03f,  -4.974033120e+03f,  4.980852240e+03f,  4.986102240e+03f,  4.977181020e+03f,
                -5.045466720e+03f,  5.052272400e+03f,  5.057539200e+03f,  5.048604540e+03f,  -5.011849920e+03f,  5.018665680e+03f,
                5.023925760e+03f,  5.014994460e+03f,  -4.978233120e+03f,  4.985058960e+03f,  4.990312320e+03f,  4.981384380e+03f,
                -5.049720480e+03f,  5.056532880e+03f,  5.061803040e+03f,  5.052861660e+03f,  -5.016076800e+03f,  5.022899280e+03f,
                5.028162720e+03f,  5.019224700e+03f,  -4.982433120e+03f,  4.989265680e+03f,  4.994522400e+03f,  4.985587740e+03f,
                -5.062481760e+03f,  5.069314320e+03f,  5.074594560e+03f,  5.065633020e+03f,  -5.028757440e+03f,  5.035600080e+03f,
                5.040873600e+03f,  5.031915420e+03f,  -4.995033120e+03f,  5.001885840e+03f,  5.007152640e+03f,  4.998197820e+03f,
                -5.066735520e+03f,  5.073574800e+03f,  5.078858400e+03f,  5.069890140e+03f,  -5.032984320e+03f,  5.039833680e+03f,
                5.045110560e+03f,  5.036145660e+03f,  -4.999233120e+03f,  5.006092560e+03f,  5.011362720e+03f,  5.002401180e+03f,
                -5.070989280e+03f,  5.077835280e+03f,  5.083122240e+03f,  5.074147260e+03f,  -5.037211200e+03f,  5.044067280e+03f,
                5.049347520e+03f,  5.040375900e+03f,  -5.003433120e+03f,  5.010299280e+03f,  5.015572800e+03f,  5.006604540e+03f,
                -5.075243040e+03f,  5.082095760e+03f,  5.087386080e+03f,  5.078404380e+03f,  -5.041438080e+03f,  5.048300880e+03f,
                5.053584480e+03f,  5.044606140e+03f,  -5.007633120e+03f,  5.014506000e+03f,  5.019782880e+03f,  5.010807900e+03f,
                -5.079496800e+03f,  5.086356240e+03f,  5.091649920e+03f,  5.082661500e+03f,  -5.045664960e+03f,  5.052534480e+03f,
                5.057821440e+03f,  5.048836380e+03f,  -5.011833120e+03f,  5.018712720e+03f,  5.023992960e+03f,  5.015011260e+03f,
                -5.092258080e+03f,  5.099137680e+03f,  5.104441440e+03f,  5.095432860e+03f,  -5.058345600e+03f,  5.065235280e+03f,
                5.070532320e+03f,  5.061527100e+03f,  -5.024433120e+03f,  5.031332880e+03f,  5.036623200e+03f,  5.027621340e+03f,
                -5.096511840e+03f,  5.103398160e+03f,  5.108705280e+03f,  5.099689980e+03f,  -5.062572480e+03f,  5.069468880e+03f,
                5.074769280e+03f,  5.065757340e+03f,  -5.028633120e+03f,  5.035539600e+03f,  5.040833280e+03f,  5.031824700e+03f,
                -5.100765600e+03f,  5.107658640e+03f,  5.112969120e+03f,  5.103947100e+03f,  -5.066799360e+03f,  5.073702480e+03f,
                5.079006240e+03f,  5.069987580e+03f,  -5.032833120e+03f,  5.039746320e+03f,  5.045043360e+03f,  5.036028060e+03f,
                -5.105019360e+03f,  5.111919120e+03f,  5.117232960e+03f,  5.108204220e+03f,  -5.071026240e+03f,  5.077936080e+03f,
                5.083243200e+03f,  5.074217820e+03f,  -5.037033120e+03f,  5.043953040e+03f,  5.049253440e+03f,  5.040231420e+03f,
                -5.109273120e+03f,  5.116179600e+03f,  5.121496800e+03f,  5.112461340e+03f,  -5.075253120e+03f,  5.082169680e+03f,
                5.087480160e+03f,  5.078448060e+03f,  -5.041233120e+03f,  5.048159760e+03f,  5.053463520e+03f,  5.044434780e+03f,
                -5.122034400e+03f,  5.128961040e+03f,  5.134288320e+03f,  5.125232700e+03f,  -5.087933760e+03f,  5.094870480e+03f,
                5.100191040e+03f,  5.091138780e+03f,  -5.053833120e+03f,  5.060779920e+03f,  5.066093760e+03f,  5.057044860e+03f,
                -5.126288160e+03f,  5.133221520e+03f,  5.138552160e+03f,  5.129489820e+03f,  -5.092160640e+03f,  5.099104080e+03f,
                5.104428000e+03f,  5.095369020e+03f,  -5.058033120e+03f,  5.064986640e+03f,  5.070303840e+03f,  5.061248220e+03f,
                -5.130541920e+03f,  5.137482000e+03f,  5.142816000e+03f,  5.133746940e+03f,  -5.096387520e+03f,  5.103337680e+03f,
                5.108664960e+03f,  5.099599260e+03f,  -5.062233120e+03f,  5.069193360e+03f,  5.074513920e+03f,  5.065451580e+03f,
                -5.134795680e+03f,  5.141742480e+03f,  5.147079840e+03f,  5.138004060e+03f,  -5.100614400e+03f,  5.107571280e+03f,
                5.112901920e+03f,  5.103829500e+03f,  -5.066433120e+03f,  5.073400080e+03f,  5.078724000e+03f,  5.069654940e+03f,
                -5.139049440e+03f,  5.146002960e+03f,  5.151343680e+03f,  5.142261180e+03f,  -5.104841280e+03f,  5.111804880e+03f,
                5.117138880e+03f,  5.108059740e+03f,  -5.070633120e+03f,  5.077606800e+03f,  5.082934080e+03f,  5.073858300e+03f,
                -5.270916000e+03f,  5.278077840e+03f,  5.283522720e+03f,  5.274231900e+03f,  -5.235874560e+03f,  5.243046480e+03f,
                5.248484640e+03f,  5.239197180e+03f,  -5.200833120e+03f,  5.208015120e+03f,  5.213446560e+03f,  5.204162460e+03f,
                -5.275169760e+03f,  5.282338320e+03f,  5.287786560e+03f,  5.278489020e+03f,  -5.240101440e+03f,  5.247280080e+03f,
                5.252721600e+03f,  5.243427420e+03f,  -5.205033120e+03f,  5.212221840e+03f,  5.217656640e+03f,  5.208365820e+03f,
                -5.279423520e+03f,  5.286598800e+03f,  5.292050400e+03f,  5.282746140e+03f,  -5.244328320e+03f,  5.251513680e+03f,
                5.256958560e+03f,  5.247657660e+03f,  -5.209233120e+03f,  5.216428560e+03f,  5.221866720e+03f,  5.212569180e+03f,
                -5.283677280e+03f,  5.290859280e+03f,  5.296314240e+03f,  5.287003260e+03f,  -5.248555200e+03f,  5.255747280e+03f,
                5.261195520e+03f,  5.251887900e+03f,  -5.213433120e+03f,  5.220635280e+03f,  5.226076800e+03f,  5.216772540e+03f,
                -5.287931040e+03f,  5.295119760e+03f,  5.300578080e+03f,  5.291260380e+03f,  -5.252782080e+03f,  5.259980880e+03f,
                5.265432480e+03f,  5.256118140e+03f,  -5.217633120e+03f,  5.224842000e+03f,  5.230286880e+03f,  5.220975900e+03f,
                -5.300692320e+03f,  5.307901200e+03f,  5.313369600e+03f,  5.304031740e+03f,  -5.265462720e+03f,  5.272681680e+03f,
                5.278143360e+03f,  5.268808860e+03f,  -5.230233120e+03f,  5.237462160e+03f,  5.242917120e+03f,  5.233585980e+03f,
                -5.304946080e+03f,  5.312161680e+03f,  5.317633440e+03f,  5.308288860e+03f,  -5.269689600e+03f,  5.276915280e+03f,
                5.282380320e+03f,  5.273039100e+03f,  -5.234433120e+03f,  5.241668880e+03f,  5.247127200e+03f,  5.237789340e+03f,
                -5.309199840e+03f,  5.316422160e+03f,  5.321897280e+03f,  5.312545980e+03f,  -5.273916480e+03f,  5.281148880e+03f,
                5.286617280e+03f,  5.277269340e+03f,  -5.238633120e+03f,  5.245875600e+03f,  5.251337280e+03f,  5.241992700e+03f,
                -5.313453600e+03f,  5.320682640e+03f,  5.326161120e+03f,  5.316803100e+03f,  -5.278143360e+03f,  5.285382480e+03f,
                5.290854240e+03f,  5.281499580e+03f,  -5.242833120e+03f,  5.250082320e+03f,  5.255547360e+03f,  5.246196060e+03f,
                -5.317707360e+03f,  5.324943120e+03f,  5.330424960e+03f,  5.321060220e+03f,  -5.282370240e+03f,  5.289616080e+03f,
                5.295091200e+03f,  5.285729820e+03f,  -5.247033120e+03f,  5.254289040e+03f,  5.259757440e+03f,  5.250399420e+03f,
                -5.330468640e+03f,  5.337724560e+03f,  5.343216480e+03f,  5.333831580e+03f,  -5.295050880e+03f,  5.302316880e+03f,
                5.307802080e+03f,  5.298420540e+03f,  -5.259633120e+03f,  5.266909200e+03f,  5.272387680e+03f,  5.263009500e+03f,
                -5.334722400e+03f,  5.341985040e+03f,  5.347480320e+03f,  5.338088700e+03f,  -5.299277760e+03f,  5.306550480e+03f,
                5.312039040e+03f,  5.302650780e+03f,  -5.263833120e+03f,  5.271115920e+03f,  5.276597760e+03f,  5.267212860e+03f,
                -5.338976160e+03f,  5.346245520e+03f,  5.351744160e+03f,  5.342345820e+03f,  -5.303504640e+03f,  5.310784080e+03f,
                5.316276000e+03f,  5.306881020e+03f,  -5.268033120e+03f,  5.275322640e+03f,  5.280807840e+03f,  5.271416220e+03f,
                -5.343229920e+03f,  5.350506000e+03f,  5.356008000e+03f,  5.346602940e+03f,  -5.307731520e+03f,  5.315017680e+03f,
                5.320512960e+03f,  5.311111260e+03f,  -5.272233120e+03f,  5.279529360e+03f,  5.285017920e+03f,  5.275619580e+03f,
                -5.347483680e+03f,  5.354766480e+03f,  5.360271840e+03f,  5.350860060e+03f,  -5.311958400e+03f,  5.319251280e+03f,
                5.324749920e+03f,  5.315341500e+03f,  -5.276433120e+03f,  5.283736080e+03f,  5.289228000e+03f,  5.279822940e+03f,
                -5.360244960e+03f,  5.367547920e+03f,  5.373063360e+03f,  5.363631420e+03f,  -5.324639040e+03f,  5.331952080e+03f,
                5.337460800e+03f,  5.328032220e+03f,  -5.289033120e+03f,  5.296356240e+03f,  5.301858240e+03f,  5.292433020e+03f,
                -5.364498720e+03f,  5.371808400e+03f,  5.377327200e+03f,  5.367888540e+03f,  -5.328865920e+03f,  5.336185680e+03f,
                5.341697760e+03f,  5.332262460e+03f,  -5.293233120e+03f,  5.300562960e+03f,  5.306068320e+03f,  5.296636380e+03f,
                -5.368752480e+03f,  5.376068880e+03f,  5.381591040e+03f,  5.372145660e+03f,  -5.333092800e+03f,  5.340419280e+03f,
                5.345934720e+03f,  5.336492700e+03f,  -5.297433120e+03f,  5.304769680e+03f,  5.310278400e+03f,  5.300839740e+03f,
                -5.373006240e+03f,  5.380329360e+03f,  5.385854880e+03f,  5.376402780e+03f,  -5.337319680e+03f,  5.344652880e+03f,
                5.350171680e+03f,  5.340722940e+03f,  -5.301633120e+03f,  5.308976400e+03f,  5.314488480e+03f,  5.305043100e+03f,
                -5.377260000e+03f,  5.384589840e+03f,  5.390118720e+03f,  5.380659900e+03f,  -5.341546560e+03f,  5.348886480e+03f,
                5.354408640e+03f,  5.344953180e+03f,  -5.305833120e+03f,  5.313183120e+03f,  5.318698560e+03f,  5.309246460e+03f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
        }
    }
}
