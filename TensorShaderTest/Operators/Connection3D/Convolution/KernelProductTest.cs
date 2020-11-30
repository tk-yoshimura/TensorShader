using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class KernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach ((int inchannels, int outchannels) in new (int, int)[] { (1, 1), (2, 1), (1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (5, 10), (10, 15), (15, 5), (15, 20), (20, 32), (32, 15), (32, 33), (33, 33) }) {
                    foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                        foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (kwidth, kheight, kdepth), (kwidth * 2, kheight * 2, kdepth * 2), (13, 13, 13), (17, 17, 17), (19, 19, 19), (17, 19, 13), (13, 17, 19), (19, 13, 17) }) {

                            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] gyval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, batch, xval);
                            Map3D gy = new Map3D(outchannels, outwidth, outheight, outdepth, batch, gyval);

                            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), gyval);

                            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth));

                            KernelProduct ope = new KernelProduct(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, batch);

                            ope.Execute(x_tensor, gy_tensor, gw_tensor);

                            float[] gw_expect = gw.ToArray();
                            float[] gw_actual = gw_tensor.State.Value;

                            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
                            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void LargeMapTest() {
            float max_err = 0;

            Random random = new Random(1234);

            int batch = 3;
            int inchannels = 49, outchannels = 50;
            int kwidth = 7, kheight = 5, kdepth = 3;
            int inwidth = 125, inheight = 196, indepth = 4;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => (float)random.NextDouble() * 1e-2f).ToArray();

            Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, batch, xval);
            Map3D gy = new Map3D(outchannels, outwidth, outheight, outdepth, batch, gyval);

            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), gyval);

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth));

            KernelProduct ope = new KernelProduct(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, batch);

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            float[] gw_expect = gw.ToArray();
            float[] gw_actual = gw_tensor.State.Value;

            CollectionAssert.AreEqual(xval, x_tensor.State.Value);
            CollectionAssert.AreEqual(gyval, gy_tensor.State.Value);

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth},{batch}");

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 32, outchannels = 32, ksize = 3;
            int outwidth = inwidth - ksize + 1, outheight = inheight - ksize + 1, outdepth = indepth - ksize + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels, outchannels, ksize, ksize, ksize));

            KernelProduct ope = new KernelProduct(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize);

            Cuda.Profiler.Initialize("../../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter3D Reference(Map3D x, Map3D gy, int kwidth, int kheight, int kdepth) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1 || outd != ind - kdepth + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter3D w = new Filter3D(inchannels, outchannels, kwidth, kheight, kdepth);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int ix, iy, iz = kz, ox, oy, oz = 0; oz < outd; iz++, oz++) {
                                for (iy = ky, oy = 0; oy < outh; iy++, oy++) {
                                    for (ix = kx, ox = 0; ox < outw; ix++, ox++) {
                                        for (int inch, outch = 0; outch < outchannels; outch++) {
                                            for (inch = 0; inch < inchannels; inch++) {
                                                w[inch, outch, kx, ky, kz] += x[inch, ix, iy, iz, th] * gy[outch, ox, oy, oz, th];
                                            }
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
            int inchannels = 2, outchannels = 3, kwidth = 3, kheight = 5, kdepth = 7, inwidth = 13, inheight = 12, indepth = 11;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outdepth * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, 1, xval);
            Map3D gy = new Map3D(outchannels, outwidth, outheight, outdepth, 1, gyval);

            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth);

            float[] gw_expect = {
                1.362072906e+02f,  1.364978943e+02f,  1.358882751e+02f,  1.361784668e+02f,  1.355692749e+02f,  1.358590393e+02f,
                1.367885437e+02f,  1.370791626e+02f,  1.364686279e+02f,  1.367588654e+02f,  1.361487579e+02f,  1.364385071e+02f,
                1.373698120e+02f,  1.376604004e+02f,  1.370490417e+02f,  1.373391418e+02f,  1.367282410e+02f,  1.370179901e+02f,
                1.437633820e+02f,  1.440540466e+02f,  1.434329376e+02f,  1.437231445e+02f,  1.431025696e+02f,  1.433922729e+02f,
                1.443446045e+02f,  1.446352539e+02f,  1.440133362e+02f,  1.443034821e+02f,  1.436819763e+02f,  1.439718018e+02f,
                1.449258728e+02f,  1.452165680e+02f,  1.445936584e+02f,  1.448838501e+02f,  1.442615204e+02f,  1.445512238e+02f,
                1.513195496e+02f,  1.516101990e+02f,  1.509776306e+02f,  1.512678680e+02f,  1.506357727e+02f,  1.509255371e+02f,
                1.519007568e+02f,  1.521913910e+02f,  1.515579987e+02f,  1.518482056e+02f,  1.512152405e+02f,  1.515049744e+02f,
                1.524819946e+02f,  1.527726440e+02f,  1.521383972e+02f,  1.524285583e+02f,  1.517947388e+02f,  1.520845032e+02f,
                1.588756561e+02f,  1.591662903e+02f,  1.585223389e+02f,  1.588124542e+02f,  1.581690674e+02f,  1.584587708e+02f,
                1.594569092e+02f,  1.597475281e+02f,  1.591026459e+02f,  1.593929138e+02f,  1.587484741e+02f,  1.590382080e+02f,
                1.600381775e+02f,  1.603287659e+02f,  1.596830597e+02f,  1.599732666e+02f,  1.593279572e+02f,  1.596176758e+02f,
                1.664317322e+02f,  1.667223816e+02f,  1.660669556e+02f,  1.663572388e+02f,  1.657021790e+02f,  1.659919739e+02f,
                1.670129700e+02f,  1.673036041e+02f,  1.666473541e+02f,  1.669375610e+02f,  1.662817688e+02f,  1.665714874e+02f,
                1.675942535e+02f,  1.678848572e+02f,  1.672277222e+02f,  1.675179443e+02f,  1.668612061e+02f,  1.671510315e+02f,
                2.268807220e+02f,  2.271713715e+02f,  2.264244080e+02f,  2.267145386e+02f,  2.259681396e+02f,  2.262579651e+02f,
                2.274619446e+02f,  2.277527161e+02f,  2.270047913e+02f,  2.272950439e+02f,  2.265476990e+02f,  2.268374634e+02f,
                2.280431976e+02f,  2.283338623e+02f,  2.275852051e+02f,  2.278753357e+02f,  2.271271667e+02f,  2.274168549e+02f,
                2.344368591e+02f,  2.347274780e+02f,  2.339691162e+02f,  2.342593384e+02f,  2.335014648e+02f,  2.337911377e+02f,
                2.350181580e+02f,  2.353086853e+02f,  2.345495605e+02f,  2.348397064e+02f,  2.340807800e+02f,  2.343705750e+02f,
                2.355993347e+02f,  2.358899689e+02f,  2.351298370e+02f,  2.354199829e+02f,  2.346604462e+02f,  2.349500885e+02f,
                2.419929199e+02f,  2.422835388e+02f,  2.415138092e+02f,  2.418039856e+02f,  2.410346680e+02f,  2.413244019e+02f,
                2.425741730e+02f,  2.428648071e+02f,  2.420941772e+02f,  2.423843079e+02f,  2.416141968e+02f,  2.419039307e+02f,
                2.431554871e+02f,  2.434460754e+02f,  2.426745300e+02f,  2.429648438e+02f,  2.421935730e+02f,  2.424833832e+02f,
                2.495490723e+02f,  2.498397217e+02f,  2.490585632e+02f,  2.493486938e+02f,  2.485679321e+02f,  2.488576813e+02f,
                2.501303406e+02f,  2.504209442e+02f,  2.496388855e+02f,  2.499290466e+02f,  2.491473999e+02f,  2.494371643e+02f,
                2.507116089e+02f,  2.510021667e+02f,  2.502191620e+02f,  2.505093994e+02f,  2.497268677e+02f,  2.500166321e+02f,
                2.571052246e+02f,  2.573958130e+02f,  2.566031494e+02f,  2.568933716e+02f,  2.561011047e+02f,  2.563908081e+02f,
                2.576864624e+02f,  2.579770508e+02f,  2.571835632e+02f,  2.574737244e+02f,  2.566806641e+02f,  2.569703979e+02f,
                2.582676392e+02f,  2.585582886e+02f,  2.577639465e+02f,  2.580540771e+02f,  2.572600708e+02f,  2.575498047e+02f,
                3.175541382e+02f,  3.178449097e+02f,  3.169606323e+02f,  3.172508850e+02f,  3.163670959e+02f,  3.166568604e+02f,
                3.181354980e+02f,  3.184260559e+02f,  3.175409546e+02f,  3.178311768e+02f,  3.169465332e+02f,  3.172362976e+02f,
                3.187166443e+02f,  3.190072632e+02f,  3.181213684e+02f,  3.184115601e+02f,  3.175260620e+02f,  3.178157959e+02f,
                3.251103516e+02f,  3.254009094e+02f,  3.245054016e+02f,  3.247955017e+02f,  3.239002075e+02f,  3.241900024e+02f,
                3.256915894e+02f,  3.259821167e+02f,  3.250856323e+02f,  3.253758545e+02f,  3.244796448e+02f,  3.247695618e+02f,
                3.262727966e+02f,  3.265634766e+02f,  3.256660767e+02f,  3.259561768e+02f,  3.250593872e+02f,  3.253490601e+02f,
                3.326665344e+02f,  3.329570007e+02f,  3.320499878e+02f,  3.323400574e+02f,  3.314335632e+02f,  3.317232971e+02f,
                3.332476501e+02f,  3.335383301e+02f,  3.326303101e+02f,  3.329205933e+02f,  3.320130615e+02f,  3.323027039e+02f,
                3.338288879e+02f,  3.341196289e+02f,  3.332106323e+02f,  3.335009460e+02f,  3.325924683e+02f,  3.328823242e+02f,
                3.402224121e+02f,  3.405131836e+02f,  3.395947266e+02f,  3.398848877e+02f,  3.389667358e+02f,  3.392565002e+02f,
                3.408038025e+02f,  3.410943909e+02f,  3.401750793e+02f,  3.404651489e+02f,  3.395462646e+02f,  3.398359985e+02f,
                3.413850708e+02f,  3.416756287e+02f,  3.407553711e+02f,  3.410455627e+02f,  3.401257324e+02f,  3.404154663e+02f,
                3.477787170e+02f,  3.480693054e+02f,  3.471394043e+02f,  3.474295959e+02f,  3.465000000e+02f,  3.467897034e+02f,
                3.483598938e+02f,  3.486505127e+02f,  3.477196960e+02f,  3.480098877e+02f,  3.470795288e+02f,  3.473691406e+02f,
                3.489411926e+02f,  3.492316895e+02f,  3.483000488e+02f,  3.485902710e+02f,  3.476589355e+02f,  3.479487305e+02f,
                4.082276306e+02f,  4.085182495e+02f,  4.074968872e+02f,  4.077869263e+02f,  4.067659302e+02f,  4.070558472e+02f,
                4.088088379e+02f,  4.090996094e+02f,  4.080770874e+02f,  4.083673706e+02f,  4.073454895e+02f,  4.076351624e+02f,
                4.093899841e+02f,  4.096807251e+02f,  4.086574707e+02f,  4.089476318e+02f,  4.079249573e+02f,  4.082147217e+02f,
                4.157837524e+02f,  4.160744629e+02f,  4.150415039e+02f,  4.153316650e+02f,  4.142990723e+02f,  4.145888062e+02f,
                4.163650513e+02f,  4.166555786e+02f,  4.156218262e+02f,  4.159121094e+02f,  4.148786316e+02f,  4.151685181e+02f,
                4.169463501e+02f,  4.172366943e+02f,  4.162023010e+02f,  4.164924011e+02f,  4.154581299e+02f,  4.157479248e+02f,
                4.233399353e+02f,  4.236305237e+02f,  4.225861816e+02f,  4.228764343e+02f,  4.218323975e+02f,  4.221221924e+02f,
                4.239210510e+02f,  4.242116089e+02f,  4.231663818e+02f,  4.234565430e+02f,  4.224119873e+02f,  4.227017212e+02f,
                4.245023804e+02f,  4.247929688e+02f,  4.237467651e+02f,  4.240371094e+02f,  4.229914246e+02f,  4.232810669e+02f,
                4.308958740e+02f,  4.311866455e+02f,  4.301307373e+02f,  4.304209900e+02f,  4.293657227e+02f,  4.296553345e+02f,
                4.314772949e+02f,  4.317677002e+02f,  4.307109070e+02f,  4.310015259e+02f,  4.299451294e+02f,  4.302348633e+02f,
                4.320582886e+02f,  4.323489380e+02f,  4.312914124e+02f,  4.315817566e+02f,  4.305245972e+02f,  4.308143921e+02f,
                4.384522705e+02f,  4.387427063e+02f,  4.376753845e+02f,  4.379657593e+02f,  4.368989868e+02f,  4.371886597e+02f,
                4.390333862e+02f,  4.393239746e+02f,  4.382557983e+02f,  4.385460205e+02f,  4.374782104e+02f,  4.377680664e+02f,
                4.396146851e+02f,  4.399050903e+02f,  4.388363037e+02f,  4.391264343e+02f,  4.380579224e+02f,  4.383475647e+02f,
                4.989009399e+02f,  4.991917419e+02f,  4.980329895e+02f,  4.983231201e+02f,  4.971648254e+02f,  4.974544983e+02f,
                4.994821777e+02f,  4.997728882e+02f,  4.986134949e+02f,  4.989034119e+02f,  4.977444458e+02f,  4.980339355e+02f,
                5.000635071e+02f,  5.003540649e+02f,  4.991937866e+02f,  4.994839478e+02f,  4.983239136e+02f,  4.986135864e+02f,
                5.064573364e+02f,  5.067478027e+02f,  5.055776672e+02f,  5.058677979e+02f,  5.046980591e+02f,  5.049877930e+02f,
                5.070385132e+02f,  5.073290710e+02f,  5.061579590e+02f,  5.064482117e+02f,  5.052774353e+02f,  5.055671387e+02f,
                5.076197815e+02f,  5.079103394e+02f,  5.067384033e+02f,  5.070285034e+02f,  5.058571777e+02f,  5.061467285e+02f,
                5.140133057e+02f,  5.143039551e+02f,  5.131221313e+02f,  5.134123535e+02f,  5.122312622e+02f,  5.125211792e+02f,
                5.145945435e+02f,  5.148851929e+02f,  5.137027588e+02f,  5.139929199e+02f,  5.128108521e+02f,  5.131005859e+02f,
                5.151757812e+02f,  5.154664307e+02f,  5.142830811e+02f,  5.145731812e+02f,  5.133901978e+02f,  5.136801147e+02f,
                5.215695801e+02f,  5.218601074e+02f,  5.206668091e+02f,  5.209571533e+02f,  5.197644043e+02f,  5.200541992e+02f,
                5.221508179e+02f,  5.224413452e+02f,  5.212474976e+02f,  5.215375977e+02f,  5.203442383e+02f,  5.206337280e+02f,
                5.227320557e+02f,  5.230227051e+02f,  5.218277588e+02f,  5.221180420e+02f,  5.209235229e+02f,  5.212131958e+02f,
                5.291255493e+02f,  5.294161377e+02f,  5.282116089e+02f,  5.285017700e+02f,  5.272979126e+02f,  5.275874634e+02f,
                5.297067871e+02f,  5.299974976e+02f,  5.287919922e+02f,  5.290822754e+02f,  5.278771362e+02f,  5.281671143e+02f,
                5.302879639e+02f,  5.305787354e+02f,  5.293723145e+02f,  5.296625366e+02f,  5.284569092e+02f,  5.287465210e+02f,
                5.895746460e+02f,  5.898652344e+02f,  5.885690918e+02f,  5.888592529e+02f,  5.875637207e+02f,  5.878535767e+02f,
                5.901558838e+02f,  5.904464722e+02f,  5.891493530e+02f,  5.894395752e+02f,  5.881430664e+02f,  5.884330444e+02f,
                5.907369995e+02f,  5.910277100e+02f,  5.897299194e+02f,  5.900198975e+02f,  5.887226562e+02f,  5.890123291e+02f,
                5.971306152e+02f,  5.974212646e+02f,  5.961137085e+02f,  5.964041748e+02f,  5.950969849e+02f,  5.953865356e+02f,
                5.977119141e+02f,  5.980026245e+02f,  5.966942139e+02f,  5.969842529e+02f,  5.956763916e+02f,  5.959661255e+02f,
                5.982929688e+02f,  5.985837402e+02f,  5.972745361e+02f,  5.975646973e+02f,  5.962561035e+02f,  5.965455933e+02f,
                6.046868896e+02f,  6.049773560e+02f,  6.036584473e+02f,  6.039487305e+02f,  6.026301880e+02f,  6.029198608e+02f,
                6.052680664e+02f,  6.055585938e+02f,  6.042387085e+02f,  6.045290527e+02f,  6.032096558e+02f,  6.034993896e+02f,
                6.058492432e+02f,  6.061398926e+02f,  6.048191528e+02f,  6.051094971e+02f,  6.037892456e+02f,  6.040788574e+02f,
                6.122429199e+02f,  6.125335083e+02f,  6.112031250e+02f,  6.114933472e+02f,  6.101633911e+02f,  6.104531250e+02f,
                6.128239746e+02f,  6.131147461e+02f,  6.117836914e+02f,  6.120737305e+02f,  6.107429810e+02f,  6.110328369e+02f,
                6.134053955e+02f,  6.136959229e+02f,  6.123638306e+02f,  6.126540527e+02f,  6.113223877e+02f,  6.116121826e+02f,
                6.197990723e+02f,  6.200895996e+02f,  6.187478638e+02f,  6.190379639e+02f,  6.176968384e+02f,  6.179864502e+02f,
                6.203802490e+02f,  6.206709595e+02f,  6.193280640e+02f,  6.196184692e+02f,  6.182762451e+02f,  6.185660400e+02f,
                6.209615479e+02f,  6.212522583e+02f,  6.199084473e+02f,  6.201987305e+02f,  6.188555298e+02f,  6.191453247e+02f,
                6.802479248e+02f,  6.805383911e+02f,  6.791053467e+02f,  6.793955078e+02f,  6.779627075e+02f,  6.782524414e+02f,
                6.808293457e+02f,  6.811198730e+02f,  6.796856689e+02f,  6.799758301e+02f,  6.785421753e+02f,  6.788319092e+02f,
                6.814103394e+02f,  6.817011719e+02f,  6.802659912e+02f,  6.805561523e+02f,  6.791215210e+02f,  6.794112549e+02f,
                6.878040771e+02f,  6.880947266e+02f,  6.866500244e+02f,  6.869403076e+02f,  6.854959106e+02f,  6.857855225e+02f,
                6.883853760e+02f,  6.886757812e+02f,  6.872304077e+02f,  6.875201416e+02f,  6.860754395e+02f,  6.863652344e+02f,
                6.889665527e+02f,  6.892570801e+02f,  6.878108521e+02f,  6.881008911e+02f,  6.866549683e+02f,  6.869445801e+02f,
                6.953602905e+02f,  6.956509399e+02f,  6.941943970e+02f,  6.944849854e+02f,  6.930291748e+02f,  6.933187866e+02f,
                6.959414673e+02f,  6.962319946e+02f,  6.947751465e+02f,  6.950651245e+02f,  6.936085815e+02f,  6.938982544e+02f,
                6.965228882e+02f,  6.968134155e+02f,  6.953554688e+02f,  6.956455078e+02f,  6.941878052e+02f,  6.944779053e+02f,
                7.029162598e+02f,  7.032069702e+02f,  7.017394409e+02f,  7.020295410e+02f,  7.005621338e+02f,  7.008519287e+02f,
                7.034975586e+02f,  7.037883301e+02f,  7.023195801e+02f,  7.026098633e+02f,  7.011418457e+02f,  7.014317627e+02f,
                7.040789185e+02f,  7.043693848e+02f,  7.029002075e+02f,  7.031903687e+02f,  7.017211304e+02f,  7.020111084e+02f,
                7.104725952e+02f,  7.107630615e+02f,  7.092839355e+02f,  7.095742188e+02f,  7.080954590e+02f,  7.083852539e+02f,
                7.110535889e+02f,  7.113445435e+02f,  7.098643799e+02f,  7.101546631e+02f,  7.086751099e+02f,  7.089648438e+02f,
                7.116349487e+02f,  7.119257202e+02f,  7.104448242e+02f,  7.107349854e+02f,  7.092546387e+02f,  7.095443115e+02f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{inwidth},{inheight},{indepth}");
        }
    }
}
