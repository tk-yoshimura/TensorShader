using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionConvolution {
    [TestClass]
    public class QuaternionConvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, inwidth = 7;
            int outwidth = inwidth - kwidth + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels, outchannels / 4, kwidth), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field y_expect = QuaternionConvolution1D(x, w);
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
            int inchannels = 8, outchannels = 12, kwidth = 3, inwidth = 7;
            int outwidth = inwidth - kwidth + 1, batch = 3;

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

            Field yr = Convolution1D(xr, wr) - Convolution1D(xi, wi) - Convolution1D(xj, wj) - Convolution1D(xk, wk);
            Field yi = Convolution1D(xr, wi) + Convolution1D(xi, wr) + Convolution1D(xj, wk) - Convolution1D(xk, wj);
            Field yj = Convolution1D(xr, wj) - Convolution1D(xi, wk) + Convolution1D(xj, wr) + Convolution1D(xk, wi);
            Field yk = Convolution1D(xr, wk) + Convolution1D(xi, wj) - Convolution1D(xj, wi) + Convolution1D(xk, wr);

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
            1.266984000e-03f,  6.381480000e-04f,  -1.182409600e-02f,  2.377752000e-03f,  3.480432000e-03f,  2.224680000e-03f, 
            -1.094444800e-02f,  2.208312000e-03f,  3.225552000e-03f,  2.052552000e-03f,  -2.436763200e-02f,  4.469220000e-03f, 
            5.884776000e-03f,  4.138524000e-03f,  -2.231203200e-02f,  4.097556000e-03f,  5.378040000e-03f,  3.759372000e-03f, 
            -3.818616000e-02f,  6.624324000e-03f,  8.262792000e-03f,  6.081084000e-03f,  -3.465830400e-02f,  6.017652000e-03f, 
            7.507224000e-03f,  5.460012000e-03f,  -5.200468800e-02f,  8.779428000e-03f,  1.064080800e-02f,  8.023644000e-03f, 
            -4.700457600e-02f,  7.937748000e-03f,  9.636408000e-03f,  7.160652000e-03f,  -2.534972800e-02f,  4.261272000e-03f, 
            5.214192000e-03f,  3.625512000e-03f,  -2.152556800e-02f,  3.621816000e-03f,  4.461648000e-03f,  2.969544000e-03f, 
            -7.033328000e-03f,  1.178124000e-03f,  1.501368000e-03f,  7.896840000e-04f,  -4.875872000e-03f,  8.192280000e-04f, 
            1.083624000e-03f,  4.213800000e-04f,  -3.786286400e-02f,  9.194028000e-03f,  1.038530400e-02f,  8.891412000e-03f, 
            -3.538515200e-02f,  8.600124000e-03f,  9.718728000e-03f,  8.281188000e-03f,  -6.841014400e-02f,  1.601301600e-02f, 
            1.806148800e-02f,  1.525284000e-02f,  -6.296396800e-02f,  1.474687200e-02f,  1.664539200e-02f,  1.395175200e-02f, 
            -8.869732800e-02f,  1.998694800e-02f,  2.253088800e-02f,  1.860044400e-02f,  -7.979193600e-02f,  1.797022800e-02f, 
            2.028232800e-02f,  1.652785200e-02f,  -1.025158560e-01f,  2.214205200e-02f,  2.490890400e-02f,  2.054300400e-02f, 
            -9.213820800e-02f,  1.989032400e-02f,  2.241151200e-02f,  1.822849200e-02f,  -1.163343840e-01f,  2.429715600e-02f, 
            2.728692000e-02f,  2.248556400e-02f,  -1.044844800e-01f,  2.181042000e-02f,  2.454069600e-02f,  1.992913200e-02f, 
            -5.453660800e-02f,  1.131631200e-02f,  1.282795200e-02f,  9.879912000e-03f,  -4.614592000e-02f,  9.580152000e-03f, 
            1.091419200e-02f,  8.094984000e-03f,  -1.477697600e-02f,  3.060588000e-03f,  3.566424000e-03f,  2.223444000e-03f, 
            -1.033625600e-02f,  2.153340000e-03f,  2.568072000e-03f,  1.290660000e-03f,  -7.300568000e-02f,  1.765671600e-02f, 
            1.941765600e-02f,  1.709893200e-02f,  -6.824470400e-02f,  1.651446000e-02f,  1.817047200e-02f,  1.592422800e-02f, 
            -1.249961920e-01f,  2.964828000e-02f,  3.264254400e-02f,  2.828100000e-02f,  -1.149834880e-01f,  2.728543200e-02f, 
            3.006523200e-02f,  2.585095200e-02f,  -1.530270240e-01f,  3.550467600e-02f,  3.917700000e-02f,  3.306236400e-02f, 
            -1.372718400e-01f,  3.184290000e-02f,  3.518661600e-02f,  2.929633200e-02f,  -1.668455520e-01f,  3.765978000e-02f, 
            4.155501600e-02f,  3.500492400e-02f,  -1.496181120e-01f,  3.376299600e-02f,  3.731580000e-02f,  3.099697200e-02f, 
            -1.806640800e-01f,  3.981488400e-02f,  4.393303200e-02f,  3.694748400e-02f,  -1.619643840e-01f,  3.568309200e-02f, 
            3.944498400e-02f,  3.269761200e-02f,  -8.372348800e-02f,  1.837135200e-02f,  2.044171200e-02f,  1.613431200e-02f, 
            -7.076627200e-02f,  1.553848800e-02f,  1.736673600e-02f,  1.322042400e-02f,  -2.252062400e-02f,  4.943052000e-03f, 
            5.631480000e-03f,  3.657204000e-03f,  -1.579664000e-02f,  3.487452000e-03f,  4.052520000e-03f,  2.159940000e-03f, 
        };

        float[] gw_expect = new float[] {
            -3.909975000e-01f,  1.288347600e-01f,  1.264263000e-01f,  1.232812800e-01f,  -4.063889400e-01f,  1.339647600e-01f, 
            1.312630200e-01f,  1.279903200e-01f,  -4.327493400e-01f,  1.047805200e-01f,  1.026408600e-01f,  9.931824000e-02f, 
            -4.500454200e-01f,  1.089428400e-01f,  1.064983800e-01f,  1.030711200e-01f,  -4.745011800e-01f,  8.072628000e-02f, 
            7.885542000e-02f,  7.535520000e-02f,  -4.937019000e-01f,  8.392092000e-02f,  8.173374000e-02f,  7.815192000e-02f, 
            -4.217803800e-01f,  1.390947600e-01f,  1.360997400e-01f,  1.326993600e-01f,  -4.371718200e-01f,  1.442247600e-01f, 
            1.409364600e-01f,  1.374084000e-01f,  -4.673415000e-01f,  1.131051600e-01f,  1.103559000e-01f,  1.068240000e-01f, 
            -4.846375800e-01f,  1.172674800e-01f,  1.142134200e-01f,  1.105768800e-01f,  -5.129026200e-01f,  8.711556000e-02f, 
            8.461206000e-02f,  8.094864000e-02f,  -5.321033400e-01f,  9.031020000e-02f,  8.749038000e-02f,  8.374536000e-02f, 
            -4.525632600e-01f,  1.493547600e-01f,  1.457731800e-01f,  1.421174400e-01f,  -4.679547000e-01f,  1.544847600e-01f, 
            1.506099000e-01f,  1.468264800e-01f,  -5.019336600e-01f,  1.214298000e-01f,  1.180709400e-01f,  1.143297600e-01f, 
            -5.192297400e-01f,  1.255921200e-01f,  1.219284600e-01f,  1.180826400e-01f,  -5.513040600e-01f,  9.350484000e-02f, 
            9.036870000e-02f,  8.654208000e-02f,  -5.705047800e-01f,  9.669948000e-02f,  9.324702000e-02f,  8.933880000e-02f, 
        };
    }
}
