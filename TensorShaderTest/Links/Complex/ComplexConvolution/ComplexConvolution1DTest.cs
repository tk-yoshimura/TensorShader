using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexConvolution {
    [TestClass]
    public class ComplexConvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, stride = 2, inwidth = 13;
            int outwidth = (inwidth - kwidth) / stride + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field y_expect = ComplexConvolution1D(x, w, stride);
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
            int inchannels = 6, outchannels = 8, kwidth = 3, stride = 2, inwidth = 13;
            int outwidth = (inwidth - kwidth) / stride + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field x_real = ComplexReal(x), x_imag = ComplexImag(x);
            Field w_real = ComplexReal(w), w_imag = ComplexImag(w);

            Field y_real = Convolution1D(x_real, w_real, stride) - Convolution1D(x_imag, w_imag, stride);
            Field y_imag = Convolution1D(x_imag, w_real, stride) + Convolution1D(x_real, w_imag, stride);

            Field y_expect = ComplexCast(y_real, y_imag);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            -7.413040000e-04f,  7.361320000e-04f,  -7.126400000e-04f,  7.129080000e-04f,  -6.839760000e-04f,  6.896840000e-04f,
            -3.973360000e-04f,  4.574440000e-04f,  -3.686720000e-04f,  4.342200000e-04f,  -3.400080000e-04f,  4.109960000e-04f,
            -2.794016000e-03f,  2.797080000e-03f,  -2.670896000e-03f,  2.690152000e-03f,  -2.547776000e-03f,  2.583224000e-03f,
            -1.607176000e-03f,  1.613876000e-03f,  -1.512720000e-03f,  1.530172000e-03f,  -1.418264000e-03f,  1.446468000e-03f,
            -5.213696000e-03f,  5.109944000e-03f,  -4.958992000e-03f,  4.882056000e-03f,  -4.704288000e-03f,  4.654168000e-03f,
            -2.817016000e-03f,  2.770308000e-03f,  -2.656768000e-03f,  2.626124000e-03f,  -2.496520000e-03f,  2.481940000e-03f,
            -7.633376000e-03f,  7.422808000e-03f,  -7.247088000e-03f,  7.073960000e-03f,  -6.860800000e-03f,  6.725112000e-03f,
            -4.026856000e-03f,  3.926740000e-03f,  -3.800816000e-03f,  3.722076000e-03f,  -3.574776000e-03f,  3.517412000e-03f,
            -1.005305600e-02f,  9.735672000e-03f,  -9.535184000e-03f,  9.265864000e-03f,  -9.017312000e-03f,  8.796056000e-03f,
            -5.236696000e-03f,  5.083172000e-03f,  -4.944864000e-03f,  4.818028000e-03f,  -4.653032000e-03f,  4.552884000e-03f,
            -1.247273600e-02f,  1.204853600e-02f,  -1.182328000e-02f,  1.145776800e-02f,  -1.117382400e-02f,  1.086700000e-02f,
            -6.446536000e-03f,  6.239604000e-03f,  -6.088912000e-03f,  5.913980000e-03f,  -5.731288000e-03f,  5.588356000e-03f,
            -2.155048000e-03f,  2.332116000e-03f,  -1.797424000e-03f,  2.006492000e-03f,  -1.439800000e-03f,  1.680868000e-03f,
            -1.176904000e-02f,  1.298638000e-02f,  -1.137672800e-02f,  1.257003600e-02f,  -1.098441600e-02f,  1.215369200e-02f,
            -7.061296000e-03f,  7.990252000e-03f,  -6.668984000e-03f,  7.573908000e-03f,  -6.276672000e-03f,  7.157564000e-03f,
            -1.612193600e-02f,  1.786269600e-02f,  -1.527152000e-02f,  1.696952800e-02f,  -1.442110400e-02f,  1.607636000e-02f,
            -8.271136000e-03f,  9.146684000e-03f,  -7.813032000e-03f,  8.669860000e-03f,  -7.354928000e-03f,  8.193036000e-03f,
            -1.854161600e-02f,  2.017556000e-02f,  -1.755961600e-02f,  1.916143200e-02f,  -1.657761600e-02f,  1.814730400e-02f,
            -9.480976000e-03f,  1.030311600e-02f,  -8.957080000e-03f,  9.765812000e-03f,  -8.433184000e-03f,  9.228508000e-03f,
            -2.096129600e-02f,  2.248842400e-02f,  -1.984771200e-02f,  2.135333600e-02f,  -1.873412800e-02f,  2.021824800e-02f,
            -1.069081600e-02f,  1.145954800e-02f,  -1.010112800e-02f,  1.086176400e-02f,  -9.511440000e-03f,  1.026398000e-02f,
            -2.338097600e-02f,  2.480128800e-02f,  -2.213580800e-02f,  2.354524000e-02f,  -2.089064000e-02f,  2.228919200e-02f,
            -1.190065600e-02f,  1.261598000e-02f,  -1.124517600e-02f,  1.195771600e-02f,  -1.058969600e-02f,  1.129945200e-02f,
            -2.580065600e-02f,  2.711415200e-02f,  -2.442390400e-02f,  2.573714400e-02f,  -2.304715200e-02f,  2.436013600e-02f,
            -1.311049600e-02f,  1.377241200e-02f,  -1.238922400e-02f,  1.305366800e-02f,  -1.166795200e-02f,  1.233492400e-02f,
            -4.455232000e-03f,  5.147484000e-03f,  -3.733960000e-03f,  4.428740000e-03f,  -3.012688000e-03f,  3.709996000e-03f,
            -2.279677600e-02f,  2.523662800e-02f,  -2.204081600e-02f,  2.442716400e-02f,  -2.128485600e-02f,  2.361770000e-02f,
            -1.372525600e-02f,  1.552306000e-02f,  -1.296929600e-02f,  1.471359600e-02f,  -1.221333600e-02f,  1.390413200e-02f,
            -2.944985600e-02f,  3.292831200e-02f,  -2.787214400e-02f,  3.124890400e-02f,  -2.629443200e-02f,  2.956949600e-02f,
            -1.493509600e-02f,  1.667949200e-02f,  -1.411334400e-02f,  1.580954800e-02f,  -1.329159200e-02f,  1.493960400e-02f,
            -3.186953600e-02f,  3.524117600e-02f,  -3.016024000e-02f,  3.344080800e-02f,  -2.845094400e-02f,  3.164044000e-02f,
            -1.614493600e-02f,  1.783592400e-02f,  -1.525739200e-02f,  1.690550000e-02f,  -1.436984800e-02f,  1.597507600e-02f,
            -3.428921600e-02f,  3.755404000e-02f,  -3.244833600e-02f,  3.563271200e-02f,  -3.060745600e-02f,  3.371138400e-02f,
            -1.735477600e-02f,  1.899235600e-02f,  -1.640144000e-02f,  1.800145200e-02f,  -1.544810400e-02f,  1.701054800e-02f,
            -3.670889600e-02f,  3.986690400e-02f,  -3.473643200e-02f,  3.782461600e-02f,  -3.276396800e-02f,  3.578232800e-02f,
            -1.856461600e-02f,  2.014878800e-02f,  -1.754548800e-02f,  1.909740400e-02f,  -1.652636000e-02f,  1.804602000e-02f,
            -3.912857600e-02f,  4.217976800e-02f,  -3.702452800e-02f,  4.001652000e-02f,  -3.492048000e-02f,  3.785327200e-02f,
            -1.977445600e-02f,  2.130522000e-02f,  -1.868953600e-02f,  2.019335600e-02f,  -1.760461600e-02f,  1.908149200e-02f,
            -6.755416000e-03f,  7.962852000e-03f,  -5.670496000e-03f,  6.850988000e-03f,  -4.585576000e-03f,  5.739124000e-03f,
        };

        float[] gw_expect = new float[] {
            -1.213749180e-01f,  2.397819600e-01f,  -1.229879700e-01f,  2.430182520e-01f,  -1.246010220e-01f,  2.462545440e-01f,
            -1.624691700e-01f,  2.065512600e-01f,  -1.646772300e-01f,  2.093326560e-01f,  -1.668852900e-01f,  2.121140520e-01f,
            -2.035634220e-01f,  1.733205600e-01f,  -2.063664900e-01f,  1.756470600e-01f,  -2.091695580e-01f,  1.779735600e-01f,
            -2.446576740e-01f,  1.400898600e-01f,  -2.480557500e-01f,  1.419614640e-01f,  -2.514538260e-01f,  1.438330680e-01f,
            -1.262140740e-01f,  2.494908360e-01f,  -1.278271260e-01f,  2.527271280e-01f,  -1.294401780e-01f,  2.559634200e-01f,
            -1.690933500e-01f,  2.148954480e-01f,  -1.713014100e-01f,  2.176768440e-01f,  -1.735094700e-01f,  2.204582400e-01f,
            -2.119726260e-01f,  1.803000600e-01f,  -2.147756940e-01f,  1.826265600e-01f,  -2.175787620e-01f,  1.849530600e-01f,
            -2.548519020e-01f,  1.457046720e-01f,  -2.582499780e-01f,  1.475762760e-01f,  -2.616480540e-01f,  1.494478800e-01f,
            -1.310532300e-01f,  2.591997120e-01f,  -1.326662820e-01f,  2.624360040e-01f,  -1.342793340e-01f,  2.656722960e-01f,
            -1.757175300e-01f,  2.232396360e-01f,  -1.779255900e-01f,  2.260210320e-01f,  -1.801336500e-01f,  2.288024280e-01f,
            -2.203818300e-01f,  1.872795600e-01f,  -2.231848980e-01f,  1.896060600e-01f,  -2.259879660e-01f,  1.919325600e-01f,
            -2.650461300e-01f,  1.513194840e-01f,  -2.684442060e-01f,  1.531910880e-01f,  -2.718422820e-01f,  1.550626920e-01f,
        };
    }
}
