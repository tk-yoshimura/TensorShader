using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.QuaternionConvolution {
    [TestClass]
    public class QuaternionDeconvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, stride = 2, inwidth = 7;
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

            Field x_expect = QuaternionDeconvolution1D(y, w, stride, Shape.Map1D(inchannels, inwidth, batch));
            Field err = x_expect - x_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gy_actual = y.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, $"not equal gw"); /*backward tolerance*/
        }

        [TestMethod]
        public void TheoreticalTest() {
            int inchannels = 8, outchannels = 12, kwidth = 3, stride = 2, inwidth = 7;
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

            Field xr = Deconvolution1D(yr, wr, stride, outshape) - Deconvolution1D(yi, wi, stride, outshape) - Deconvolution1D(yj, wj, stride, outshape) - Deconvolution1D(yk, wk, stride, outshape);
            Field xi = Deconvolution1D(yr, wi, stride, outshape) + Deconvolution1D(yi, wr, stride, outshape) + Deconvolution1D(yj, wk, stride, outshape) - Deconvolution1D(yk, wj, stride, outshape);
            Field xj = Deconvolution1D(yr, wj, stride, outshape) - Deconvolution1D(yi, wk, stride, outshape) + Deconvolution1D(yj, wr, stride, outshape) + Deconvolution1D(yk, wi, stride, outshape);
            Field xk = Deconvolution1D(yr, wk, stride, outshape) + Deconvolution1D(yi, wj, stride, outshape) - Deconvolution1D(yj, wi, stride, outshape) + Deconvolution1D(yk, wr, stride, outshape);

            Field x_expect = QuaternionCast(xr, xi, xj, xk);

            Field err = x_expect - x_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gy_actual = y.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, $"not equal gw"); /*backward tolerance*/
        }

        float[] gy_expect = new float[] {
            -7.898256000e-03f,  4.520640000e-04f,  1.256656000e-03f,  1.142720000e-04f,  -5.949200000e-03f,  2.776000000e-04f,
            9.424160000e-04f,  -3.561600000e-05f,  -4.000144000e-03f,  1.031360000e-04f,  6.281760000e-04f,  -1.855040000e-04f,
            -2.254896000e-02f,  2.320884000e-03f,  3.340808000e-03f,  1.833916000e-03f,  -1.793075200e-02f,  1.772372000e-03f,
            2.630056000e-03f,  1.295004000e-03f,  -1.331254400e-02f,  1.223860000e-03f,  1.919304000e-03f,  7.560920000e-04f,
            -3.805310400e-02f,  3.147264000e-03f,  4.431472000e-03f,  2.464640000e-03f,  -3.032408000e-02f,  2.659456000e-03f,
            3.785456000e-03f,  2.006016000e-03f,  -2.259505600e-02f,  2.171648000e-03f,  3.139440000e-03f,  1.547392000e-03f,
            -6.036436800e-02f,  5.819424000e-03f,  7.268368000e-03f,  4.795808000e-03f,  -4.885217600e-02f,  4.539040000e-03f,
            5.792912000e-03f,  3.512352000e-03f,  -3.733998400e-02f,  3.258656000e-03f,  4.317456000e-03f,  2.228896000e-03f,
            -7.426728000e-02f,  8.290884000e-03f,  1.005192800e-02f,  7.166476000e-03f,  -6.017579200e-02f,  6.567332000e-03f,
            8.097016000e-03f,  5.417964000e-03f,  -4.608430400e-02f,  4.843780000e-03f,  6.142104000e-03f,  3.669452000e-03f,
            -9.072657600e-02f,  8.058432000e-03f,  1.015288000e-02f,  6.772928000e-03f,  -7.310264000e-02f,  6.796480000e-03f,
            8.677424000e-03f,  5.512512000e-03f,  -5.547870400e-02f,  5.534528000e-03f,  7.201968000e-03f,  4.252096000e-03f,
            -1.128304800e-01f,  1.118678400e-02f,  1.328008000e-02f,  9.477344000e-03f,  -9.175515200e-02f,  8.800480000e-03f,
            1.064340800e-02f,  7.060320000e-03f,  -7.067982400e-02f,  6.414176000e-03f,  8.006736000e-03f,  4.643296000e-03f,
            -1.259856000e-01f,  1.426088400e-02f,  1.676304800e-02f,  1.249903600e-02f,  -1.024208320e-01f,  1.136229200e-02f,
            1.356397600e-02f,  9.540924000e-03f,  -7.885606400e-02f,  8.463700000e-03f,  1.036490400e-02f,  6.582812000e-03f,
            -1.434000480e-01f,  1.296960000e-02f,  1.587428800e-02f,  1.108121600e-02f,  -1.158812000e-01f,  1.093350400e-02f,
            1.356939200e-02f,  9.019008000e-03f,  -8.836235200e-02f,  8.897408000e-03f,  1.126449600e-02f,  6.956800000e-03f,
        };

        float[] gw_expect = new float[] {
            -1.558111500e-01f,  2.752806000e-02f,  2.773301400e-02f,  2.668617600e-02f,  -1.655163900e-01f,  2.494966800e-02f,
            2.529804600e-02f,  2.418036000e-02f,  -1.647940380e-01f,  2.915149200e-02f,  2.920827000e-02f,  2.813464800e-02f,
            -1.752181260e-01f,  2.642449200e-02f,  2.662181400e-02f,  2.548310400e-02f,  -1.737769260e-01f,  3.077492400e-02f,
            3.068352600e-02f,  2.958312000e-02f,  -1.849198620e-01f,  2.789931600e-02f,  2.794558200e-02f,  2.678584800e-02f,
            -1.831605300e-01f,  1.469293200e-02f,  1.543991400e-02f,  1.450252800e-02f,  -1.918504260e-01f,  1.313449200e-02f,
            1.402720200e-02f,  1.301436000e-02f,  -1.940281380e-01f,  1.556953200e-02f,  1.616689800e-02f,  1.524535200e-02f,
            -2.033838900e-01f,  1.391778000e-02f,  1.465914600e-02f,  1.366560000e-02f,  -2.048957460e-01f,  1.644613200e-02f,
            1.689388200e-02f,  1.598817600e-02f,  -2.149173540e-01f,  1.470106800e-02f,  1.529109000e-02f,  1.431684000e-02f,
            -1.909238220e-01f,  2.081938800e-02f,  2.154072600e-02f,  2.032771200e-02f,  -2.006256060e-01f,  1.825827600e-02f,
            1.912995000e-02f,  1.783226400e-02f,  -2.024353500e-01f,  2.220090000e-02f,  2.276369400e-02f,  2.153426400e-02f,
            -2.128698060e-01f,  1.947735600e-02f,  2.018760600e-02f,  1.887926400e-02f,  -2.139468780e-01f,  2.358241200e-02f,
            2.398666200e-02f,  2.274081600e-02f,  -2.251140060e-01f,  2.069643600e-02f,  2.124526200e-02f,  1.992626400e-02f,
        };
    }
}
