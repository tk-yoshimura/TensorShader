using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionConvolution {
    [TestClass]
    public class QuaternionDeconvolution1DTest {
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

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field x_expect = QuaternionDeconvolution1D(y, w);
            Field err = x_expect - x_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gy_actual = y.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, $"not equal gw"); /*backward tolerance*/
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

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field yr = QuaternionR(y), yi = QuaternionI(y), yj = QuaternionJ(y), yk = QuaternionK(y);
            Field wr = QuaternionR(w), wi = QuaternionI(w), wj = QuaternionJ(w), wk = QuaternionK(w);

            Shape outshape = Shape.Map1D(inchannels / 4, inwidth, batch);

            Field xr = Deconvolution1D(yr, wr) - Deconvolution1D(yi, wi) - Deconvolution1D(yj, wj) - Deconvolution1D(yk, wk);
            Field xi = Deconvolution1D(yr, wi) + Deconvolution1D(yi, wr) + Deconvolution1D(yj, wk) - Deconvolution1D(yk, wj);
            Field xj = Deconvolution1D(yr, wj) - Deconvolution1D(yi, wk) + Deconvolution1D(yj, wr) + Deconvolution1D(yk, wi);
            Field xk = Deconvolution1D(yr, wk) + Deconvolution1D(yi, wj) - Deconvolution1D(yj, wi) + Deconvolution1D(yk, wr);

            Field x_expect = QuaternionCast(xr, xi, xj, xk);

            Field err = x_expect - x_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gy_actual = y.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, $"not equal gw"); /*backward tolerance*/
        }

        float[] gy_expect = new float[] {
            -6.367968000e-03f,  2.133912000e-03f,  2.970768000e-03f,  1.929000000e-03f,  -4.853664000e-03f,  1.533912000e-03f,
            2.195280000e-03f,  1.307496000e-03f,  -3.339360000e-03f,  9.339120000e-04f,  1.419792000e-03f,  6.859920000e-04f,
            -1.154832000e-02f,  5.290848000e-03f,  6.282464000e-03f,  5.127424000e-03f,  -9.151648000e-03f,  4.076384000e-03f,
            4.852192000e-03f,  3.856896000e-03f,  -6.754976000e-03f,  2.861920000e-03f,  3.421920000e-03f,  2.586368000e-03f,
            -1.604817600e-02f,  9.026868000e-03f,  1.015624800e-02f,  8.819484000e-03f,  -1.285934400e-02f,  7.125012000e-03f,
            7.997688000e-03f,  6.837372000e-03f,  -9.670512000e-03f,  5.223156000e-03f,  5.839128000e-03f,  4.855260000e-03f,
            -2.203824000e-02f,  1.101456000e-02f,  1.237673600e-02f,  1.073977600e-02f,  -1.733449600e-02f,  9.127328000e-03f,
            1.022761600e-02f,  8.787264000e-03f,  -1.263075200e-02f,  7.240096000e-03f,  8.078496000e-03f,  6.834752000e-03f,
            -3.160905600e-02f,  9.375384000e-03f,  1.091649600e-02f,  8.929320000e-03f,  -2.479862400e-02f,  8.056536000e-03f,
            9.366864000e-03f,  7.588968000e-03f,  -1.798819200e-02f,  6.737688000e-03f,  7.817232000e-03f,  6.248616000e-03f,
            -5.016556800e-02f,  1.633567200e-02f,  1.770628800e-02f,  1.538964000e-02f,  -4.100966400e-02f,  1.283263200e-02f,
            1.388952000e-02f,  1.179597600e-02f,  -3.185376000e-02f,  9.329592000e-03f,  1.007275200e-02f,  8.202312000e-03f,
            -5.102160000e-02f,  2.354044800e-02f,  2.525014400e-02f,  2.272806400e-02f,  -4.155932800e-02f,  1.891606400e-02f,
            2.022563200e-02f,  1.795545600e-02f,  -3.209705600e-02f,  1.429168000e-02f,  1.520112000e-02f,  1.318284800e-02f,
            -5.427513600e-02f,  2.828086800e-02f,  3.028960800e-02f,  2.750516400e-02f,  -4.417046400e-02f,  2.285389200e-02f,
            2.439856800e-02f,  2.189425200e-02f,  -3.406579200e-02f,  1.742691600e-02f,  1.850752800e-02f,  1.628334000e-02f,
            -6.185712000e-02f,  2.850384000e-02f,  3.086057600e-02f,  2.771833600e-02f,  -4.953481600e-02f,  2.375964800e-02f,
            2.567017600e-02f,  2.281670400e-02f,  -3.721251200e-02f,  1.901545600e-02f,  2.047977600e-02f,  1.791507200e-02f,
            -7.609785600e-02f,  2.205650400e-02f,  2.468433600e-02f,  2.114580000e-02f,  -6.053990400e-02f,  1.894053600e-02f,
            2.119934400e-02f,  1.793920800e-02f,  -4.498195200e-02f,  1.582456800e-02f,  1.771435200e-02f,  1.473261600e-02f,
            -9.396316800e-02f,  3.053743200e-02f,  3.244180800e-02f,  2.885028000e-02f,  -7.716566400e-02f,  2.413135200e-02f,
            2.558376000e-02f,  2.228445600e-02f,  -6.036816000e-02f,  1.772527200e-02f,  1.872571200e-02f,  1.571863200e-02f,
            -9.049488000e-02f,  4.179004800e-02f,  4.421782400e-02f,  4.032870400e-02f,  -7.396700800e-02f,  3.375574400e-02f,
            3.559907200e-02f,  3.205401600e-02f,  -5.743913600e-02f,  2.572144000e-02f,  2.698032000e-02f,  2.377932800e-02f,
            -9.250209600e-02f,  4.753486800e-02f,  5.042296800e-02f,  4.619084400e-02f,  -7.548158400e-02f,  3.858277200e-02f,
            4.079944800e-02f,  3.695113200e-02f,  -5.846107200e-02f,  2.963067600e-02f,  3.117592800e-02f,  2.771142000e-02f,
            -1.016760000e-01f,  4.599312000e-02f,  4.934441600e-02f,  4.469689600e-02f,  -8.173513600e-02f,  3.839196800e-02f,
            4.111273600e-02f,  3.684614400e-02f,  -6.179427200e-02f,  3.079081600e-02f,  3.288105600e-02f,  2.899539200e-02f,
            -1.205866560e-01f,  3.473762400e-02f,  3.845217600e-02f,  3.336228000e-02f,  -9.628118400e-02f,  2.982453600e-02f,
            3.303182400e-02f,  2.828944800e-02f,  -7.197571200e-02f,  2.491144800e-02f,  2.761147200e-02f,  2.321661600e-02f,
        };

        float[] gw_expect = new float[] {
            -3.329735280e-01f,  1.857254400e-01f,  1.806313680e-01f,  1.764937440e-01f,  -3.711545520e-01f,  1.680436800e-01f,
            1.631792400e-01f,  1.589423520e-01f,  -3.443181360e-01f,  1.920139200e-01f,  1.866107280e-01f,  1.823010720e-01f,
            -3.840328560e-01f,  1.737446400e-01f,  1.685641680e-01f,  1.641690720e-01f,  -3.556627440e-01f,  1.983024000e-01f,
            1.925900880e-01f,  1.881084000e-01f,  -3.969111600e-01f,  1.794456000e-01f,  1.739490960e-01f,  1.693957920e-01f,
            -3.777614700e-01f,  1.812485400e-01f,  1.760614140e-01f,  1.716721920e-01f,  -4.174861020e-01f,  1.620588840e-01f,
            1.571192460e-01f,  1.525950480e-01f,  -3.907642620e-01f,  1.877739240e-01f,  1.822550700e-01f,  1.776750960e-01f,
            -4.321078380e-01f,  1.679103480e-01f,  1.626314940e-01f,  1.579315200e-01f,  -4.037670540e-01f,  1.942993080e-01f,
            1.884487260e-01f,  1.836780000e-01f,  -4.467295740e-01f,  1.737618120e-01f,  1.681437420e-01f,  1.632679920e-01f,
            -4.526628720e-01f,  1.475346240e-01f,  1.430817360e-01f,  1.391147040e-01f,  -4.908266160e-01f,  1.299392640e-01f,
            1.257505680e-01f,  1.216151520e-01f,  -4.683900720e-01f,  1.532355840e-01f,  1.484424720e-01f,  1.443137760e-01f,
            -5.081566320e-01f,  1.349835840e-01f,  1.304477520e-01f,  1.261644960e-01f,  -4.841172720e-01f,  1.589365440e-01f,
            1.538032080e-01f,  1.495128480e-01f,  -5.254866480e-01f,  1.400279040e-01f,  1.351449360e-01f,  1.307138400e-01f,
        };
    }
}
