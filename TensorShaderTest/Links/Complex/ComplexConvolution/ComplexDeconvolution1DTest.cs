using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexConvolution {
    [TestClass]
    public class ComplexDeconvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, stride = 2, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field x_expect = ComplexDeconvolution1D(y, w, stride, Shape.Map1D(inchannels, inwidth, batch));
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
            int inchannels = 6, outchannels = 8, kwidth = 3, stride = 2, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field y_real = ComplexReal(y), y_imag = ComplexImag(y);
            Field w_real = ComplexReal(w), w_imag = ComplexImag(w);

            Field x_real = Deconvolution1D(y_real, w_real, stride, Shape.Map1D(inchannels / 2, inwidth, batch))
                         - Deconvolution1D(y_imag, w_imag, stride, Shape.Map1D(inchannels / 2, inwidth, batch));

            Field x_imag = Deconvolution1D(y_imag, w_real, stride, Shape.Map1D(inchannels / 2, inwidth, batch))
                         + Deconvolution1D(y_real, w_imag, stride, Shape.Map1D(inchannels / 2, inwidth, batch));

            Field x_expect = ComplexCast(x_real, x_imag);

            Field err = x_expect - x_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gy_actual = y.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, $"not equal gw"); /*backward tolerance*/
        }

        float[] gy_expect = new float[] {
            -4.339456000e-03f,  3.895440000e-04f,  -3.554800000e-03f,  2.894640000e-04f,  -2.770144000e-03f,  1.893840000e-04f,
            -1.985488000e-03f,  8.930400000e-05f,  -1.254987200e-02f,  1.699260000e-03f,  -1.066584800e-02f,  1.406004000e-03f,
            -8.781824000e-03f,  1.112748000e-03f,  -6.897800000e-03f,  8.194920000e-04f,  -2.063819200e-02f,  3.106300000e-03f,
            -1.766552800e-02f,  2.611444000e-03f,  -1.469286400e-02f,  2.116588000e-03f,  -1.172020000e-02f,  1.621732000e-03f,
            -2.872651200e-02f,  4.513340000e-03f,  -2.466520800e-02f,  3.816884000e-03f,  -2.060390400e-02f,  3.120428000e-03f,
            -1.654260000e-02f,  2.423972000e-03f,  -3.681483200e-02f,  5.920380000e-03f,  -3.166488800e-02f,  5.022324000e-03f,
            -2.651494400e-02f,  4.124268000e-03f,  -2.136500000e-02f,  3.226212000e-03f,  -4.636816000e-02f,  5.785224000e-03f,
            -3.969102400e-02f,  5.125272000e-03f,  -3.301388800e-02f,  4.465320000e-03f,  -2.633675200e-02f,  3.805368000e-03f,
            -5.861905600e-02f,  7.899768000e-03f,  -5.057161600e-02f,  6.666120000e-03f,  -4.252417600e-02f,  5.432472000e-03f,
            -3.447673600e-02f,  4.198824000e-03f,  -6.588579200e-02f,  1.008750000e-02f,  -5.682192800e-02f,  8.584644000e-03f,
            -4.775806400e-02f,  7.081788000e-03f,  -3.869420000e-02f,  5.578932000e-03f,  -7.397411200e-02f,  1.149454000e-02f,
            -6.382160800e-02f,  9.790084000e-03f,  -5.366910400e-02f,  8.085628000e-03f,  -4.351660000e-02f,  6.381172000e-03f,
            -8.206243200e-02f,  1.290158000e-02f,  -7.082128800e-02f,  1.099552400e-02f,  -5.958014400e-02f,  9.089468000e-03f,
            -4.833900000e-02f,  7.183412000e-03f,  -9.015075200e-02f,  1.430862000e-02f,  -7.782096800e-02f,  1.220096400e-02f,
            -6.549118400e-02f,  1.009330800e-02f,  -5.316140000e-02f,  7.985652000e-03f,  -1.010901280e-01f,  1.274248800e-02f,
            -8.681843200e-02f,  1.128074400e-02f,  -7.254673600e-02f,  9.819000000e-03f,  -5.827504000e-02f,  8.357256000e-03f,
            -1.128986560e-01f,  1.540999200e-02f,  -9.758843200e-02f,  1.304277600e-02f,  -8.227820800e-02f,  1.067556000e-02f,
            -6.696798400e-02f,  8.308344000e-03f,  -1.192217120e-01f,  1.847574000e-02f,  -1.029780080e-01f,  1.576328400e-02f,
            -8.673430400e-02f,  1.305082800e-02f,  -7.049060000e-02f,  1.033837200e-02f,  -1.273100320e-01f,  1.988278000e-02f,
            -1.099776880e-01f,  1.696872400e-02f,  -9.264534400e-02f,  1.405466800e-02f,  -7.531300000e-02f,  1.114061200e-02f,
            -1.353983520e-01f,  2.128982000e-02f,  -1.169773680e-01f,  1.817416400e-02f,  -9.855638400e-02f,  1.505850800e-02f,
            -8.013540000e-02f,  1.194285200e-02f,  -1.434866720e-01f,  2.269686000e-02f,  -1.239770480e-01f,  1.937960400e-02f,
            -1.044674240e-01f,  1.606234800e-02f,  -8.495780000e-02f,  1.274509200e-02f,  -1.558120960e-01f,  1.969975200e-02f,
            -1.339458400e-01f,  1.743621600e-02f,  -1.120795840e-01f,  1.517268000e-02f,  -9.021332800e-02f,  1.290914400e-02f,
        };

        float[] gw_expect = new float[] {
            -2.976504120e-01f,  6.882074400e-02f,  -3.060629160e-01f,  6.537590400e-02f,  -3.144754200e-01f,  6.193106400e-02f,
            -3.039771720e-01f,  7.025414400e-02f,  -3.126067320e-01f,  6.673519200e-02f,  -3.212362920e-01f,  6.321624000e-02f,
            -3.103039320e-01f,  7.168754400e-02f,  -3.191505480e-01f,  6.809448000e-02f,  -3.279971640e-01f,  6.450141600e-02f,
            -3.166306920e-01f,  7.312094400e-02f,  -3.256943640e-01f,  6.945376800e-02f,  -3.347580360e-01f,  6.578659200e-02f,
            -3.462147600e-01f,  3.594782400e-02f,  -3.530539440e-01f,  3.407750400e-02f,  -3.598931280e-01f,  3.220718400e-02f,
            -3.536859120e-01f,  3.667488000e-02f,  -3.607099920e-01f,  3.476308800e-02f,  -3.677340720e-01f,  3.285129600e-02f,
            -3.611570640e-01f,  3.740193600e-02f,  -3.683660400e-01f,  3.544867200e-02f,  -3.755750160e-01f,  3.349540800e-02f,
            -3.686282160e-01f,  3.812899200e-02f,  -3.760220880e-01f,  3.613425600e-02f,  -3.834159600e-01f,  3.413952000e-02f,
            -3.353607480e-01f,  6.085274400e-02f,  -3.437694120e-01f,  5.741558400e-02f,  -3.521780760e-01f,  5.397842400e-02f,
            -3.426421320e-01f,  6.219129600e-02f,  -3.512716920e-01f,  5.867618400e-02f,  -3.599012520e-01f,  5.516107200e-02f,
            -3.499235160e-01f,  6.352984800e-02f,  -3.587739720e-01f,  5.993678400e-02f,  -3.676244280e-01f,  5.634372000e-02f,
            -3.572049000e-01f,  6.486840000e-02f,  -3.662762520e-01f,  6.119738400e-02f,  -3.753476040e-01f,  5.752636800e-02f,
        };
    }
}
