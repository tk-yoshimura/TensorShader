using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionConvolution {
    [TestClass]
    public class QuaternionConvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, stride = 2, inwidth = 7;
            int outwidth = (inwidth - kwidth) / stride + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field y_expect = QuaternionConvolution1D(x, w, stride);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, stride = 2, inwidth = 7;
            int outwidth = (inwidth - kwidth) / stride + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field xr = QuaternionR(x), xi = QuaternionI(x), xj = QuaternionJ(x), xk = QuaternionK(x);
            Field wr = QuaternionR(w), wi = QuaternionI(w), wj = QuaternionJ(w), wk = QuaternionK(w);

            Field yr = Convolution1D(xr, wr, stride) - Convolution1D(xi, wi, stride) - Convolution1D(xj, wj, stride) - Convolution1D(xk, wk, stride);
            Field yi = Convolution1D(xr, wi, stride) + Convolution1D(xi, wr, stride) + Convolution1D(xj, wk, stride) - Convolution1D(xk, wj, stride);
            Field yj = Convolution1D(xr, wj, stride) - Convolution1D(xi, wk, stride) + Convolution1D(xj, wr, stride) + Convolution1D(xk, wi, stride);
            Field yk = Convolution1D(xr, wk, stride) + Convolution1D(xi, wj, stride) - Convolution1D(xj, wi, stride) + Convolution1D(xk, wr, stride);

            Field y_expect = QuaternionCast(yr, yi, yj, yk);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            -2.720048000e-03f,  7.313400000e-04f,  1.352952000e-03f,  6.838920000e-04f,  -2.525600000e-03f,  6.857880000e-04f,
            1.266984000e-03f,  6.381480000e-04f,  -1.553360000e-03f,  4.580280000e-04f,  8.371440000e-04f,  4.094280000e-04f,
            -1.358912000e-03f,  4.124760000e-04f,  7.511760000e-04f,  3.636840000e-04f,  -9.352096000e-03f,  3.364824000e-03f,
            4.254960000e-03f,  3.225576000e-03f,  -8.557696000e-03f,  3.117048000e-03f,  3.917136000e-03f,  2.972808000e-03f,
            -5.365712000e-03f,  1.966764000e-03f,  2.422488000e-03f,  1.848468000e-03f,  -4.765760000e-03f,  1.764540000e-03f,
            2.170632000e-03f,  1.641444000e-03f,  -1.697680000e-02f,  6.382296000e-03f,  7.425648000e-03f,  6.103656000e-03f,
            -1.537139200e-02f,  5.821176000e-03f,  6.756048000e-03f,  5.528328000e-03f,  -9.178064000e-03f,  3.475500000e-03f,
            4.007832000e-03f,  3.287508000e-03f,  -8.172608000e-03f,  3.116604000e-03f,  3.590088000e-03f,  2.919204000e-03f,
            -3.145328000e-03f,  1.322124000e-03f,  1.501368000e-03f,  1.077684000e-03f,  -2.139872000e-03f,  9.632280000e-04f,
            1.083624000e-03f,  7.093800000e-04f,  -2.015086400e-02f,  9.338028000e-03f,  1.038530400e-02f,  9.179412000e-03f,
            -1.882515200e-02f,  8.744124000e-03f,  9.718728000e-03f,  8.569188000e-03f,  -1.219659200e-02f,  5.774604000e-03f,
            6.385848000e-03f,  5.518068000e-03f,  -1.087088000e-02f,  5.180700000e-03f,  5.719272000e-03f,  4.907844000e-03f,
            -3.063856000e-02f,  1.399797600e-02f,  1.535236800e-02f,  1.344285600e-02f,  -2.758163200e-02f,  1.265349600e-02f,
            1.385332800e-02f,  1.206112800e-02f,  -1.600894400e-02f,  7.283340000e-03f,  7.971192000e-03f,  6.957108000e-03f,
            -1.427772800e-02f,  6.532764000e-03f,  7.138728000e-03f,  6.185604000e-03f,  -3.826326400e-02f,  1.701544800e-02f,
            1.852305600e-02f,  1.632093600e-02f,  -3.439532800e-02f,  1.535762400e-02f,  1.669224000e-02f,  1.461664800e-02f,
            -1.982129600e-02f,  8.792076000e-03f,  9.556536000e-03f,  8.396148000e-03f,  -1.768457600e-02f,  7.884828000e-03f,
            8.558184000e-03f,  7.463364000e-03f,  -7.000976000e-03f,  3.348588000e-03f,  3.566424000e-03f,  2.799444000e-03f,
            -4.864256000e-03f,  2.441340000e-03f,  2.568072000e-03f,  1.866660000e-03f,  -3.758168000e-02f,  1.794471600e-02f,
            1.941765600e-02f,  1.767493200e-02f,  -3.512470400e-02f,  1.680246000e-02f,  1.817047200e-02f,  1.650022800e-02f,
            -2.283982400e-02f,  1.109118000e-02f,  1.193455200e-02f,  1.062670800e-02f,  -2.038284800e-02f,  9.948924000e-03f,
            1.068736800e-02f,  9.452004000e-03f,  -5.192502400e-02f,  2.463112800e-02f,  2.644977600e-02f,  2.366013600e-02f,
            -4.660556800e-02f,  2.218994400e-02f,  2.378952000e-02f,  2.114944800e-02f,  -2.665217600e-02f,  1.259991600e-02f,
            1.351989600e-02f,  1.206574800e-02f,  -2.378969600e-02f,  1.130098800e-02f,  1.210682400e-02f,  1.072976400e-02f,
            -5.954972800e-02f,  2.764860000e-02f,  2.962046400e-02f,  2.653821600e-02f,  -5.341926400e-02f,  2.489407200e-02f,
            2.662843200e-02f,  2.370496800e-02f,  -3.046452800e-02f,  1.410865200e-02f,  1.510524000e-02f,  1.350478800e-02f,
            -2.719654400e-02f,  1.265305200e-02f,  1.352628000e-02f,  1.200752400e-02f,  -1.085662400e-02f,  5.375052000e-03f,
            5.631480000e-03f,  4.521204000e-03f,  -7.588640000e-03f,  3.919452000e-03f,  4.052520000e-03f,  3.023940000e-03f,
        };

        float[] gw_expect = new float[] {
            -1.043677800e-01f,  7.770636000e-02f,  7.495606800e-02f,  7.370784000e-02f,  -1.084186440e-01f,  8.078436000e-02f,
            7.785810000e-02f,  7.653326400e-02f,  -1.294926120e-01f,  6.320008800e-02f,  6.061107600e-02f,  5.925628800e-02f,
            -1.346862600e-01f,  6.569748000e-02f,  6.292558800e-02f,  6.150801600e-02f,  -1.546174440e-01f,  4.869381600e-02f,
            4.626608400e-02f,  4.480473600e-02f,  -1.609538760e-01f,  5.061060000e-02f,  4.799307600e-02f,  4.648276800e-02f,
            -1.124695080e-01f,  8.386236000e-02f,  8.076013200e-02f,  7.935868800e-02f,  -1.165203720e-01f,  8.694036000e-02f,
            8.366216400e-02f,  8.218411200e-02f,  -1.398799080e-01f,  6.819487200e-02f,  6.524010000e-02f,  6.375974400e-02f,
            -1.450735560e-01f,  7.069226400e-02f,  6.755461200e-02f,  6.601147200e-02f,  -1.672903080e-01f,  5.252738400e-02f,
            4.972006800e-02f,  4.816080000e-02f,  -1.736267400e-01f,  5.444416800e-02f,  5.144706000e-02f,  4.983883200e-02f,
            -1.205712360e-01f,  9.001836000e-02f,  8.656419600e-02f,  8.500953600e-02f,  -1.246221000e-01f,  9.309636000e-02f,
            8.946622800e-02f,  8.783496000e-02f,  -1.502672040e-01f,  7.318965600e-02f,  6.986912400e-02f,  6.826320000e-02f,
            -1.554608520e-01f,  7.568704800e-02f,  7.218363600e-02f,  7.051492800e-02f,  -1.799631720e-01f,  5.636095200e-02f,
            5.317405200e-02f,  5.151686400e-02f,  -1.862996040e-01f,  5.827773600e-02f,  5.490104400e-02f,  5.319489600e-02f,
        };
    }
}
