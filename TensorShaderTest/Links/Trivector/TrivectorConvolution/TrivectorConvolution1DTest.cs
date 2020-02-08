using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorConvolution {
    [TestClass]
    public class TrivectorConvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, kwidth = 3, inwidth = 7;
            int outwidth = inwidth - kwidth + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            ParameterField x = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            ParameterField w = new Tensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);
            VariableField y_actual = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);

            Field y_expect = TrivectorConvolution1D(x, w);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-6f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int inchannels = 9, outchannels = 12, kwidth = 3, inwidth = 7;
            int outwidth = inwidth - kwidth + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            ParameterField x = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            ParameterField w = new Tensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);
            VariableField y_actual = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);

            Field xx = TrivectorX(x), xy = TrivectorY(x), xz = TrivectorZ(x);
            Field wr = QuaternionR(w), wi = QuaternionI(w), wj = QuaternionJ(w), wk = QuaternionK(w);
            Field wrr = wr * wr, wii = wi * wi, wjj = wj * wj, wkk = wk * wk;
            Field wri = wr * wi, wrj = wr * wj, wrk = wr * wk, wij = wi * wj, wik = wi * wk, wjk = wj * wk;

            Field yx = Convolution1D(xx, (wrr + wii - wjj - wkk)) + 2 * (Convolution1D(xy, (wij - wrk)) + Convolution1D(xz, (wik + wrj)));
            Field yy = Convolution1D(xy, (wrr - wii + wjj - wkk)) + 2 * (Convolution1D(xz, (wjk - wri)) + Convolution1D(xx, (wij + wrk)));
            Field yz = Convolution1D(xz, (wrr - wii - wjj + wkk)) + 2 * (Convolution1D(xx, (wik - wrj)) + Convolution1D(xy, (wjk + wri)));

            Field y_expect = TrivectorCast(yx, yy, yz);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            -8.049368383e-04f,  -9.929351354e-04f,  -3.970793001e-04f,  -7.478715623e-04f,  -9.241621379e-04f,  -3.651065321e-04f,
            -6.929044060e-04f,  -8.578685373e-04f,  -3.344324214e-04f,  -3.471195103e-03f,  -3.732373228e-03f,  -2.828627555e-03f,
            -3.230152896e-03f,  -3.472497125e-03f,  -2.631161796e-03f,  -2.998328503e-03f,  -3.222600892e-03f,  -2.441279093e-03f,
            -6.807258827e-03f,  -7.081744027e-03f,  -6.021461079e-03f,  -6.315586923e-03f,  -6.568687627e-03f,  -5.584811013e-03f,
            -5.845274101e-03f,  -6.078132648e-03f,  -5.167014146e-03f,  -1.027231642e-02f,  -1.054838505e-02f,  -9.356129343e-03f,
            -9.505153909e-03f,  -9.759838788e-03f,  -8.651653669e-03f,  -8.774415204e-03f,  -9.008857180e-03f,  -7.980988421e-03f,
            -1.373737401e-02f,  -1.401502606e-02f,  -1.269079761e-02f,  -1.269472089e-02f,  -1.295098995e-02f,  -1.171849632e-02f,
            -1.170355631e-02f,  -1.193958171e-02f,  -1.079496270e-02f,  -4.359350558e-03f,  -4.448038279e-03f,  -3.966232619e-03f,
            -3.858717572e-03f,  -3.937699655e-03f,  -3.503599051e-03f,  -3.397431845e-03f,  -3.467467360e-03f,  -3.078462993e-03f,
            -5.882316940e-04f,  -6.025645288e-04f,  -5.081503488e-04f,  -4.552360428e-04f,  -4.669889160e-04f,  -3.895601769e-04f,
            -3.444248078e-04f,  -3.539770059e-04f,  -2.922116316e-04f,  -1.168541072e-02f,  -1.187582627e-02f,  -1.089286283e-02f,
            -1.093850581e-02f,  -1.111725718e-02f,  -1.018733729e-02f,  -1.021656203e-02f,  -1.038402702e-02f,  -9.505722600e-03f,
            -1.846302645e-02f,  -1.872933171e-02f,  -1.722996628e-02f,  -1.711666119e-02f,  -1.736417785e-02f,  -1.595674133e-02f,
            -1.582523979e-02f,  -1.605472292e-02f,  -1.473632382e-02f,  -2.243360596e-02f,  -2.271571916e-02f,  -2.099428293e-02f,
            -2.069548382e-02f,  -2.095621775e-02f,  -1.935101361e-02f,  -1.904730982e-02f,  -1.928779637e-02f,  -1.779443407e-02f,
            -2.589866355e-02f,  -2.618236018e-02f,  -2.432895119e-02f,  -2.388505081e-02f,  -2.414736891e-02f,  -2.241785626e-02f,
            -2.197645093e-02f,  -2.221852090e-02f,  -2.060840835e-02f,  -2.936372114e-02f,  -2.964900120e-02f,  -2.766361946e-02f,
            -2.707461779e-02f,  -2.733852007e-02f,  -2.548469892e-02f,  -2.490559203e-02f,  -2.514924543e-02f,  -2.342238262e-02f,
            -9.105223808e-03f,  -9.199122274e-03f,  -8.443270936e-03f,  -8.047980229e-03f,  -8.132134735e-03f,  -7.447570885e-03f,
            -7.075809945e-03f,  -7.150972597e-03f,  -6.534592739e-03f,  -1.222747479e-03f,  -1.239581177e-03f,  -1.079633474e-03f,
            -9.486246449e-04f,  -9.628383191e-04f,  -8.301832392e-04f,  -7.195492458e-04f,  -7.315186994e-04f,  -6.245868276e-04f,
            -2.256588460e-02f,  -2.275871741e-02f,  -2.138864637e-02f,  -2.112914005e-02f,  -2.131035223e-02f,  -2.000956805e-02f,
            -1.974021965e-02f,  -1.991018551e-02f,  -1.867701278e-02f,  -3.345485780e-02f,  -3.372629020e-02f,  -3.163130500e-02f,
            -3.100316949e-02f,  -3.125585857e-02f,  -2.928232086e-02f,  -2.865215107e-02f,  -2.888684495e-02f,  -2.703136855e-02f,
            -3.805995309e-02f,  -3.834969429e-02f,  -3.596710478e-02f,  -3.507538072e-02f,  -3.534374788e-02f,  -3.311721620e-02f,
            -3.224934555e-02f,  -3.249746009e-02f,  -3.042185400e-02f,  -4.152501068e-02f,  -4.181633531e-02f,  -3.930177304e-02f,
            -3.826494771e-02f,  -3.853489904e-02f,  -3.618405885e-02f,  -3.517848665e-02f,  -3.542818463e-02f,  -3.323582827e-02f,
            -4.499006827e-02f,  -4.528297633e-02f,  -4.263644131e-02f,  -4.145451469e-02f,  -4.172605020e-02f,  -3.925090151e-02f,
            -3.810762775e-02f,  -3.835890916e-02f,  -3.604980255e-02f,  -1.385109706e-02f,  -1.395020627e-02f,  -1.292030925e-02f,
            -1.223724288e-02f,  -1.232656982e-02f,  -1.139154272e-02f,  -1.075418804e-02f,  -1.083447783e-02f,  -9.990722485e-03f,
            -1.857263264e-03f,  -1.876597825e-03f,  -1.651116599e-03f,  -1.442013247e-03f,  -1.458687722e-03f,  -1.270806301e-03f,
            -1.094673684e-03f,  -1.109060393e-03f,  -9.569620237e-04f
        };

        float[] gw_expect = new float[] {
            -8.036744546e-02f,  -7.571618778e-02f,  -7.922717298e-02f,  -8.270205712e-02f,  -8.016705921e-02f,  -7.549216164e-02f,
            -7.898090516e-02f,  -8.248429093e-02f,  -7.984880354e-02f,  -7.515396650e-02f,  -7.861768397e-02f,  -8.214403885e-02f,
            -8.522171649e-02f,  -8.060575520e-02f,  -8.393341133e-02f,  -8.714252286e-02f,  -8.481704609e-02f,  -8.017971012e-02f,
            -8.347659893e-02f,  -8.670893975e-02f,  -8.427393149e-02f,  -7.961853807e-02f,  -8.288236413e-02f,  -8.613257342e-02f,
            -8.678331767e-02f,  -8.222460962e-02f,  -8.536051228e-02f,  -8.831368691e-02f,  -8.612162815e-02f,  -8.154296897e-02f,
            -8.464150461e-02f,  -8.761289373e-02f,  -8.530302476e-02f,  -8.070737835e-02f,  -8.376669571e-02f,  -8.675111020e-02f,
            -8.540552823e-02f,  -8.092549281e-02f,  -8.386202380e-02f,  -8.656909723e-02f,  -8.444355765e-02f,  -7.994428735e-02f,
            -8.283857604e-02f,  -8.555910670e-02f,  -8.330830867e-02f,  -7.879244390e-02f,  -8.164303839e-02f,  -8.437200887e-02f,
            -5.758770070e-02f,  -5.368195817e-02f,  -5.632457976e-02f,  -5.905553897e-02f,  -5.650329380e-02f,  -5.260166455e-02f,
            -5.520116163e-02f,  -5.791912862e-02f,  -5.530101748e-02f,  -5.140720193e-02f,  -5.396079012e-02f,  -5.666023238e-02f,
            -5.830241882e-02f,  -5.441842902e-02f,  -5.686649006e-02f,  -5.934767158e-02f,  -5.685941693e-02f,  -5.297893162e-02f,
            -5.537900966e-02f,  -5.784321434e-02f,  -5.527797085e-02f,  -5.140430725e-02f,  -5.375410687e-02f,  -5.619597388e-02f,
            -5.595427588e-02f,  -5.211142587e-02f,  -5.436234701e-02f,  -5.660435917e-02f,  -5.411573243e-02f,  -5.027516039e-02f,
            -5.247483008e-02f,  -5.469613831e-02f,  -5.212027512e-02f,  -4.828494494e-02f,  -5.043151192e-02f,  -5.262692708e-02f,
            -5.092497021e-02f,  -4.714251267e-02f,  -4.919391612e-02f,  -5.120736725e-02f,  -4.866341166e-02f,  -4.488152222e-02f,
            -4.687979425e-02f,  -4.886907189e-02f,  -4.622857471e-02f,  -4.244989378e-02f,  -4.439358249e-02f,  -4.635366923e-02f,
            -3.056465681e-02f,  -2.753764466e-02f,  -2.921166501e-02f,  -3.099952887e-02f,  -2.859622926e-02f,  -2.560108355e-02f,
            -2.721109656e-02f,  -2.894447437e-02f,  -2.650993229e-02f,  -2.355035345e-02f,  -2.509357474e-02f,  -2.676693398e-02f,
            -2.639913002e-02f,  -2.336653170e-02f,  -2.485236241e-02f,  -2.641262448e-02f,  -2.391779666e-02f,  -2.091358198e-02f,
            -2.233421401e-02f,  -2.383729311e-02f,  -2.129801909e-02f,  -1.832550529e-02f,  -1.967864322e-02f,  -2.111917852e-02f,
            -1.947633524e-02f,  -1.645604295e-02f,  -1.775533728e-02f,  -1.909937853e-02f,  -1.646093787e-02f,  -1.346515266e-02f,
            -1.469931109e-02f,  -1.598372998e-02f,  -1.328862664e-02f,  -1.032031238e-02f,  -1.148748367e-02f,  -1.270709107e-02f,
            -1.020638991e-02f,  -7.216564582e-03f,  -8.330572713e-03f,  -9.469774079e-03f,  -6.645243379e-03f,  -3.675789149e-03f,
            -4.725776729e-03f,  -5.803173894e-03f,  -2.910818453e-03f,  3.562428216e-05f,  -9.488908645e-04f,  -1.959466398e-03f,
        };
    }
}
