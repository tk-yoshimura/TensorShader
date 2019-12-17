using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderCudaBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class KernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            TestCaseIterator iterator = new TestCaseIterator(
                times: 5,
                new int[] { 1, 2 },
                new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 },
                new int[] { 1, 2, 3, 4, 5, 10, 15, 20, 32, 33 },
                new int[] { 1, 3, 5 },
                new int[] { 1, 3, 5 },
                new int[] { 1, 3, 5 },
                new int[] { 5, 7, 12, 13, 17, 19 }, 
                new int[] { 5, 7, 12, 13, 17, 19 },
                new int[] { 5, 7, 12, 13, 17, 19 }
            );

            foreach(int[] testcase in iterator) {  
                int batch = testcase[0], inchannels = testcase[1], outchannels = testcase[2];
                int kwidth = testcase[3], kheight = testcase[4], kdepth = testcase[5];
                int inwidth = testcase[6], inheight = testcase[7], indepth = testcase[8];
                int stride = 1;

                int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                float[] gyval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, batch, xval);
                Map3D gy = new Map3D(outchannels, outwidth, outheight, outdepth, batch, gyval);

                Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth, stride);

                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch), xval);
                OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch), gyval);

                OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels, outchannels, kwidth, kheight, kdepth));

                KernelProduct ope = new KernelProduct(inwidth, inheight, indepth, inchannels, outchannels, kwidth, kheight, kdepth, stride, batch);

                ope.Execute(x_tensor, gy_tensor, gw_tensor);

                float[] gw_expect = gw.ToArray();
                float[] gw_actual = gw_tensor.State;

                CollectionAssert.AreEqual(xval, x_tensor.State);
                CollectionAssert.AreEqual(gyval, gy_tensor.State);

                AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");

                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, inchannels = 32, outchannels = 32, ksize = 3, stride = 1;
            int outwidth = (inwidth - ksize) / stride + 1, outheight = (inheight - ksize) / stride + 1, outdepth = (indepth - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(inchannels, inwidth, inheight, indepth));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map3D(outchannels, outwidth, outheight, outdepth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel3D(inchannels, outchannels, ksize, ksize, ksize));

            KernelProduct ope = new KernelProduct(inwidth, inheight, indepth, inchannels, outchannels, ksize, ksize, ksize, stride);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/kernelproduct_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            Cuda.Profiler.Stop();
        }

        public static Filter3D Reference(Map3D x, Map3D gy, int kwidth, int kheight, int kdepth, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1 || outd != (ind - kdepth) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter3D w = new Filter3D(inchannels, outchannels, kwidth, kheight, kdepth);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int inch, outch = 0; outch < outchannels; outch++) {
                                for (inch = 0; inch < inchannels; inch++) {
                                    double sum = 0;

                                    for (int ix, iy, iz = kz, ox, oy, oz = 0; oz < outd; iz += stride, oz++) {
                                        for (iy = ky, oy = 0; oy < outh; iy += stride, oy++) {
                                            for (ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                                                sum += x[inch, ix, iy, iz, th] * gy[outch, ox, oy, oz, th];
                                            }
                                        }
                                    }

                                    w[inch, outch, kx, ky, kz] += sum;
                                }
                            }
                        }
                    }
                }
            }

            return w;
        }

        public static Filter3D OptimizedReference(Map3D x, Map3D gy, int kwidth, int kheight, int kdepth, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = gy.Width, outh = gy.Height, outd = gy.Depth;

            if (outw != inw - kwidth + 1 || outh != inh - kheight + 1 || outd != (ind - kdepth) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter3D w = new Filter3D(inchannels, outchannels, kwidth, kheight, kdepth);

            for (int kx, ky, kz = 0; kz < kdepth; kz++) {
                for (ky = 0; ky < kheight; ky++) {
                    for (kx = 0; kx < kwidth; kx++) {
                        for (int th = 0; th < batch; th++) {
                            for (int outch, inch = 0; inch < inchannels; inch++) {
                                for (outch = 0; outch < outchannels; outch++) {
                                    int filter_idx = inch + inchannels * outch + (kx + ky * kwidth + kz * kwidth * kheight) * inchannels * outchannels;
                                    int inmap_org = inch + (kx + ky * inw + kz * inw * inh) * inchannels + th * inw * inh * ind * inchannels;
                                    int outmap_idx = outch + th * outw * outh * outd * outchannels;

                                    double sum = 0;

                                    for (int ox, oy, oz = 0; oz < outd; oz++) {
                                        int inmap_car = inmap_org;

                                        for (oy = 0; oy < outh; oy++) {
                                            int inmap_idx = inmap_car;

                                            for (ox = 0; ox < outw; ox++) {
                                                sum += x[inmap_idx] * gy[outmap_idx];

                                                inmap_idx += inchannels * stride;
                                                outmap_idx += outchannels;
                                            }

                                            inmap_car += inchannels * inw * stride;
                                        }

                                        inmap_org += inchannels * inw * inh * stride;
                                    }

                                    w[filter_idx] += sum;
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
            int inchannels = 2, outchannels = 3, kwidth = 3, kheight = 5, kdepth = 7, stride = 2, inwidth = 13, inheight = 12, indepth = 11;
            int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

            float[] xval = (new float[inwidth * inheight * indepth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outdepth * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, 1, xval);
            Map3D gy = new Map3D(outchannels, outwidth, outheight, outdepth, 1, gyval);

            Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth, stride);

            float[] gw_expect = {
                3.318840504e+00f,  3.326652765e+00f,  3.267576694e+00f,  3.275316000e+00f,  3.216312170e+00f,  3.223979712e+00f,
                3.334464073e+00f,  3.342275858e+00f,  3.283056498e+00f,  3.290796041e+00f,  3.231648207e+00f,  3.239315748e+00f,
                3.350088120e+00f,  3.357900143e+00f,  3.298536301e+00f,  3.306276321e+00f,  3.246984005e+00f,  3.254652500e+00f,
                3.521952391e+00f,  3.529764891e+00f,  3.468816519e+00f,  3.476556063e+00f,  3.415679932e+00f,  3.423348188e+00f,
                3.537576437e+00f,  3.545387983e+00f,  3.484296083e+00f,  3.492036104e+00f,  3.431016445e+00f,  3.438684464e+00f,
                3.553200006e+00f,  3.561012030e+00f,  3.499776125e+00f,  3.507516384e+00f,  3.446352005e+00f,  3.454020023e+00f,
                3.725064039e+00f,  3.732876301e+00f,  3.670056820e+00f,  3.677796125e+00f,  3.615047693e+00f,  3.622716188e+00f,
                3.740688324e+00f,  3.748500347e+00f,  3.685536385e+00f,  3.693275928e+00f,  3.630384445e+00f,  3.638051510e+00f,
                3.756312132e+00f,  3.764124393e+00f,  3.701015949e+00f,  3.708756208e+00f,  3.645719290e+00f,  3.653388500e+00f,
                3.928175688e+00f,  3.935988665e+00f,  3.871296406e+00f,  3.879036188e+00f,  3.814416647e+00f,  3.822084188e+00f,
                3.943800449e+00f,  3.951611996e+00f,  3.886776209e+00f,  3.894516230e+00f,  3.829752445e+00f,  3.837419271e+00f,
                3.959424257e+00f,  3.967235804e+00f,  3.902256966e+00f,  3.909996510e+00f,  3.845088243e+00f,  3.852756023e+00f,
                4.131289005e+00f,  4.139100075e+00f,  4.072535992e+00f,  4.080276966e+00f,  4.013784885e+00f,  4.021452427e+00f,
                4.146911621e+00f,  4.154724121e+00f,  4.088016510e+00f,  4.095757008e+00f,  4.029120445e+00f,  4.036788464e+00f,
                4.162535667e+00f,  4.170348167e+00f,  4.103495598e+00f,  4.111236095e+00f,  4.044456005e+00f,  4.052124977e+00f,
                5.756184578e+00f,  5.763996601e+00f,  5.682456017e+00f,  5.690197468e+00f,  5.608727455e+00f,  5.616396427e+00f,
                5.771809101e+00f,  5.779620647e+00f,  5.697938442e+00f,  5.705675602e+00f,  5.624063969e+00f,  5.631732941e+00f,
                5.787433147e+00f,  5.795242310e+00f,  5.713416576e+00f,  5.721157551e+00f,  5.639401913e+00f,  5.647068501e+00f,
                5.959295750e+00f,  5.967109203e+00f,  5.883696079e+00f,  5.891435146e+00f,  5.808095932e+00f,  5.815765381e+00f,
                5.974919796e+00f,  5.982733250e+00f,  5.899176121e+00f,  5.906915188e+00f,  5.823432446e+00f,  5.831099987e+00f,
                5.990544796e+00f,  5.998356342e+00f,  5.914656639e+00f,  5.922397137e+00f,  5.838769913e+00f,  5.846436024e+00f,
                6.162408352e+00f,  6.170218468e+00f,  6.084935665e+00f,  6.092675209e+00f,  6.007464886e+00f,  6.015132904e+00f,
                6.178032398e+00f,  6.185844898e+00f,  6.100416183e+00f,  6.108155727e+00f,  6.022800446e+00f,  6.030468941e+00f,
                6.193657875e+00f,  6.201467514e+00f,  6.115896225e+00f,  6.123636246e+00f,  6.038137436e+00f,  6.045805931e+00f,
                6.365520954e+00f,  6.373331547e+00f,  6.286175728e+00f,  6.293916225e+00f,  6.206833363e+00f,  6.214501381e+00f,
                6.381144524e+00f,  6.388956070e+00f,  6.301656723e+00f,  6.309396267e+00f,  6.222167969e+00f,  6.229837894e+00f,
                6.396769524e+00f,  6.404580116e+00f,  6.317136765e+00f,  6.324876785e+00f,  6.237503529e+00f,  6.245172024e+00f,
                6.568633556e+00f,  6.576446056e+00f,  6.487416267e+00f,  6.495157242e+00f,  6.406201363e+00f,  6.413867950e+00f,
                6.584257603e+00f,  6.592068672e+00f,  6.502896786e+00f,  6.510635853e+00f,  6.421535969e+00f,  6.429205418e+00f,
                6.599881649e+00f,  6.607692719e+00f,  6.518376827e+00f,  6.526116848e+00f,  6.436872482e+00f,  6.444540501e+00f,
                8.193530083e+00f,  8.201340675e+00f,  8.097335815e+00f,  8.105076790e+00f,  8.001144409e+00f,  8.008813858e+00f,
                8.209151268e+00f,  8.216964722e+00f,  8.112816811e+00f,  8.120556831e+00f,  8.016480446e+00f,  8.024147987e+00f,
                8.224775314e+00f,  8.232589722e+00f,  8.128297806e+00f,  8.136035919e+00f,  8.031816483e+00f,  8.039483070e+00f,
                8.396640778e+00f,  8.404454231e+00f,  8.298578262e+00f,  8.306315422e+00f,  8.200512886e+00f,  8.208180428e+00f,
                8.412264824e+00f,  8.420076370e+00f,  8.314057350e+00f,  8.321797371e+00f,  8.215850830e+00f,  8.223517418e+00f,
                8.427887917e+00f,  8.435700417e+00f,  8.329539299e+00f,  8.337276459e+00f,  8.231184959e+00f,  8.238853455e+00f,
                8.599752426e+00f,  8.607564926e+00f,  8.499815941e+00f,  8.507555008e+00f,  8.399881363e+00f,  8.407549858e+00f,
                8.615378380e+00f,  8.623187065e+00f,  8.515295982e+00f,  8.523036003e+00f,  8.415216446e+00f,  8.422885895e+00f,
                8.631002426e+00f,  8.638812065e+00f,  8.530778885e+00f,  8.538516045e+00f,  8.430552483e+00f,  8.438220978e+00f,
                8.802865028e+00f,  8.810678482e+00f,  8.701053619e+00f,  8.708798409e+00f,  8.599247932e+00f,  8.606918335e+00f,
                8.818490982e+00f,  8.826302528e+00f,  8.716535568e+00f,  8.724276543e+00f,  8.614583969e+00f,  8.622253418e+00f,
                8.834112167e+00f,  8.841923714e+00f,  8.732017517e+00f,  8.739757538e+00f,  8.629920959e+00f,  8.637588501e+00f,
                9.005975723e+00f,  9.013788223e+00f,  8.902297974e+00f,  8.910037994e+00f,  8.798616409e+00f,  8.806283951e+00f,
                9.021600723e+00f,  9.029413223e+00f,  8.917778015e+00f,  8.925515175e+00f,  8.813950539e+00f,  8.821619987e+00f,
                9.037225723e+00f,  9.045037270e+00f,  8.933253288e+00f,  8.940998077e+00f,  8.829288483e+00f,  8.836956024e+00f,
                1.063086987e+01f,  1.063868427e+01f,  1.051221848e+01f,  1.051995659e+01f,  1.039355659e+01f,  1.040122795e+01f,
                1.064649391e+01f,  1.065430832e+01f,  1.052769756e+01f,  1.053544044e+01f,  1.040889835e+01f,  1.041656399e+01f,
                1.066212368e+01f,  1.066993237e+01f,  1.054317856e+01f,  1.055091763e+01f,  1.042423534e+01f,  1.043190098e+01f,
                1.083398819e+01f,  1.084179688e+01f,  1.071345901e+01f,  1.072119904e+01f,  1.059293079e+01f,  1.060059452e+01f,
                1.084960938e+01f,  1.085741901e+01f,  1.072893620e+01f,  1.073667717e+01f,  1.060826683e+01f,  1.061593246e+01f,
                1.086523247e+01f,  1.087304497e+01f,  1.074441814e+01f,  1.075215721e+01f,  1.062360287e+01f,  1.063126945e+01f,
                1.103709698e+01f,  1.104490852e+01f,  1.091469479e+01f,  1.092243767e+01f,  1.079229546e+01f,  1.079996490e+01f,
                1.105272198e+01f,  1.106053543e+01f,  1.093017769e+01f,  1.093791580e+01f,  1.080763054e+01f,  1.081530094e+01f,
                1.106834793e+01f,  1.107615948e+01f,  1.094565868e+01f,  1.095339680e+01f,  1.082296944e+01f,  1.083063984e+01f,
                1.124020481e+01f,  1.124802017e+01f,  1.111593819e+01f,  1.112367630e+01f,  1.099166775e+01f,  1.099933529e+01f,
                1.125583363e+01f,  1.126364517e+01f,  1.113141823e+01f,  1.113915920e+01f,  1.100699902e+01f,  1.101467133e+01f,
                1.127145481e+01f,  1.127927208e+01f,  1.114689636e+01f,  1.115463924e+01f,  1.102233791e+01f,  1.103000546e+01f,
                1.144331646e+01f,  1.145113087e+01f,  1.131717777e+01f,  1.132491589e+01f,  1.119103336e+01f,  1.119870186e+01f,
                1.145894527e+01f,  1.146675873e+01f,  1.133265877e+01f,  1.134039783e+01f,  1.120637035e+01f,  1.121403790e+01f,
                1.147457123e+01f,  1.148238087e+01f,  1.134813786e+01f,  1.135587883e+01f,  1.122170067e+01f,  1.122936916e+01f,
                1.306821918e+01f,  1.307603073e+01f,  1.292709541e+01f,  1.293484020e+01f,  1.278597736e+01f,  1.279364395e+01f,
                1.308384037e+01f,  1.309165478e+01f,  1.294257450e+01f,  1.295031643e+01f,  1.280131435e+01f,  1.280898380e+01f,
                1.309946251e+01f,  1.310727596e+01f,  1.295805836e+01f,  1.296579552e+01f,  1.281664944e+01f,  1.282431793e+01f,
                1.327132893e+01f,  1.327914333e+01f,  1.312833500e+01f,  1.313607979e+01f,  1.298534489e+01f,  1.299301243e+01f,
                1.328695011e+01f,  1.329476738e+01f,  1.314381886e+01f,  1.315155506e+01f,  1.300068378e+01f,  1.300834846e+01f,
                1.330257702e+01f,  1.331039047e+01f,  1.315929699e+01f,  1.316703320e+01f,  1.301601601e+01f,  1.302368450e+01f,
                1.347444439e+01f,  1.348225594e+01f,  1.332957935e+01f,  1.333731842e+01f,  1.318471432e+01f,  1.319238281e+01f,
                1.349006367e+01f,  1.349787807e+01f,  1.334505749e+01f,  1.335279655e+01f,  1.320004845e+01f,  1.320771694e+01f,
                1.350569153e+01f,  1.351350307e+01f,  1.336054134e+01f,  1.336827469e+01f,  1.321538639e+01f,  1.322305202e+01f,
                1.367755508e+01f,  1.368536568e+01f,  1.353081799e+01f,  1.353855801e+01f,  1.338407898e+01f,  1.339174938e+01f,
                1.369317818e+01f,  1.370098591e+01f,  1.354629612e+01f,  1.355403423e+01f,  1.339941978e+01f,  1.340708733e+01f,
                1.370879936e+01f,  1.371661377e+01f,  1.356178188e+01f,  1.356952000e+01f,  1.341475487e+01f,  1.342242241e+01f,
                1.388066292e+01f,  1.388848019e+01f,  1.373205853e+01f,  1.373979855e+01f,  1.358344936e+01f,  1.359111500e+01f,
                1.389629364e+01f,  1.390409851e+01f,  1.374753761e+01f,  1.375527668e+01f,  1.359878922e+01f,  1.360645390e+01f,
                1.391191483e+01f,  1.391972542e+01f,  1.376301861e+01f,  1.377075672e+01f,  1.361411953e+01f,  1.362178993e+01f,
                1.550556183e+01f,  1.551337433e+01f,  1.534197903e+01f,  1.534971428e+01f,  1.517839146e+01f,  1.518606186e+01f,
                1.552118206e+01f,  1.552899933e+01f,  1.535745811e+01f,  1.536520100e+01f,  1.519372845e+01f,  1.520140171e+01f,
                1.553680992e+01f,  1.554461861e+01f,  1.537293816e+01f,  1.538067722e+01f,  1.520906353e+01f,  1.521673489e+01f,
                1.570867348e+01f,  1.571648788e+01f,  1.554321861e+01f,  1.555095863e+01f,  1.537776279e+01f,  1.538542747e+01f,
                1.572429657e+01f,  1.573211479e+01f,  1.555869389e+01f,  1.556644154e+01f,  1.539309692e+01f,  1.540076351e+01f,
                1.573992443e+01f,  1.574773598e+01f,  1.557417583e+01f,  1.558191872e+01f,  1.540843487e+01f,  1.541610336e+01f,
                1.591178799e+01f,  1.591959953e+01f,  1.574445724e+01f,  1.575219917e+01f,  1.557713127e+01f,  1.558479500e+01f,
                1.592740917e+01f,  1.593521881e+01f,  1.575993919e+01f,  1.576767540e+01f,  1.559246540e+01f,  1.560013294e+01f,
                1.594303513e+01f,  1.595084286e+01f,  1.577541924e+01f,  1.578315830e+01f,  1.560780144e+01f,  1.561546803e+01f,
                1.611489487e+01f,  1.612271118e+01f,  1.594569778e+01f,  1.595343876e+01f,  1.577649975e+01f,  1.578416634e+01f,
                1.613052177e+01f,  1.613833427e+01f,  1.596117687e+01f,  1.596891880e+01f,  1.579183483e+01f,  1.579949665e+01f,
                1.614614487e+01f,  1.615395546e+01f,  1.597665882e+01f,  1.598439503e+01f,  1.580716801e+01f,  1.581483936e+01f,
                1.631801033e+01f,  1.632582474e+01f,  1.614693642e+01f,  1.615467644e+01f,  1.597586727e+01f,  1.598353004e+01f,
                1.633363342e+01f,  1.634144783e+01f,  1.616241837e+01f,  1.617015648e+01f,  1.599120426e+01f,  1.599886703e+01f,
                1.634925461e+01f,  1.635706902e+01f,  1.617789650e+01f,  1.618563652e+01f,  1.600653839e+01f,  1.601420784e+01f,
                1.794290733e+01f,  1.795071983e+01f,  1.775685883e+01f,  1.776459885e+01f,  1.757081223e+01f,  1.757847977e+01f,
                1.795852852e+01f,  1.796634102e+01f,  1.777233315e+01f,  1.778007889e+01f,  1.758614922e+01f,  1.759381676e+01f,
                1.797415543e+01f,  1.798196602e+01f,  1.778781509e+01f,  1.779555511e+01f,  1.760148430e+01f,  1.760914803e+01f,
                1.814601517e+01f,  1.815382957e+01f,  1.795809364e+01f,  1.796583748e+01f,  1.777018166e+01f,  1.777784920e+01f,
                1.816164017e+01f,  1.816946030e+01f,  1.797357559e+01f,  1.798131752e+01f,  1.778550911e+01f,  1.779318237e+01f,
                1.817727280e+01f,  1.818508148e+01f,  1.798905373e+01f,  1.799679375e+01f,  1.780084801e+01f,  1.780851936e+01f,
                1.834912682e+01f,  1.835693932e+01f,  1.815933609e+01f,  1.816707993e+01f,  1.796954727e+01f,  1.797721100e+01f,
                1.836475563e+01f,  1.837257004e+01f,  1.817481804e+01f,  1.818255997e+01f,  1.798488426e+01f,  1.799254608e+01f,
                1.838037682e+01f,  1.838818932e+01f,  1.819029808e+01f,  1.819803429e+01f,  1.800021744e+01f,  1.800788307e+01f,
                1.855224419e+01f,  1.856005478e+01f,  1.836058235e+01f,  1.836831665e+01f,  1.816892242e+01f,  1.817658424e+01f,
                1.856786728e+01f,  1.857567787e+01f,  1.837605476e+01f,  1.838380051e+01f,  1.818424797e+01f,  1.819191933e+01f,
                1.858349228e+01f,  1.859129906e+01f,  1.839154053e+01f,  1.839928246e+01f,  1.819958305e+01f,  1.820725250e+01f,
                1.875535202e+01f,  1.876316643e+01f,  1.856181908e+01f,  1.856954956e+01f,  1.836828423e+01f,  1.837594986e+01f,
                1.877097702e+01f,  1.877878952e+01f,  1.857729530e+01f,  1.858504105e+01f,  1.838361549e+01f,  1.839128685e+01f,
                1.878660011e+01f,  1.879441452e+01f,  1.859277725e+01f,  1.860051537e+01f,  1.839895248e+01f,  1.840662003e+01f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        foreach ((int kwidth, int kheight, int kdepth) in new (int, int, int)[] { (1, 1, 1), (3, 3, 3), (5, 5, 5), (1, 3, 5), (3, 5, 1), (5, 1, 3) }) {
                            foreach ((int stride, int inwidth, int inheight, int indepth) in new (int, int, int, int)[] { (1, 13, 13, 13), (2, 17, 17, 17), (3, 19, 19, 19), (1, 17, 19, 13), (2, 13, 17, 19), (3, 19, 13, 17) }) {
                                int outwidth = inwidth - kwidth + 1, outheight = inheight - kheight + 1, outdepth = indepth - kdepth + 1;

                                float[] xval = (new float[inwidth * inheight * indepth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] gyval = (new float[outwidth * outheight * outdepth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map3D x = new Map3D(inchannels, inwidth, inheight, indepth, batch, xval);
                                Map3D gy = new Map3D(outchannels, outwidth, outheight, outdepth, batch, gyval);

                                Filter3D gw = Reference(x, gy, kwidth, kheight, kdepth, stride);
                                Filter3D gw_optimized = OptimizedReference(x, gy, kwidth, kheight, kdepth, stride);

                                float[] gw_expect = gw.ToArray();
                                float[] gw_actual = gw_optimized.ToArray();

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{kdepth},{stride},{inwidth},{inheight},{indepth},{batch}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
