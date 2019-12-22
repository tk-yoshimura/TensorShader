using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexConvolution {
    [TestClass]
    public class ComplexDeconvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, inwidth = 13;
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

            Field x_expect = ComplexDeconvolution1D(y, w);
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
            int inchannels = 6, outchannels = 8, kwidth = 3, inwidth = 13;
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

            Field x_real = Deconvolution1D(y_real, w_real)
                         - Deconvolution1D(y_imag, w_imag);

            Field x_imag = Deconvolution1D(y_imag, w_real)
                         + Deconvolution1D(y_real, w_imag);

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
            -3.243136000e-03f,  1.582064000e-03f,  -2.675488000e-03f,  1.255760000e-03f,  -2.107840000e-03f,  9.294560000e-04f, 
            -1.540192000e-03f,  6.031520000e-04f,  -5.861200000e-03f,  3.788584000e-03f,  -4.966240000e-03f,  3.142744000e-03f, 
            -4.071280000e-03f,  2.496904000e-03f,  -3.176320000e-03f,  1.851064000e-03f,  -8.046672000e-03f,  6.364332000e-03f, 
            -6.866232000e-03f,  5.364036000e-03f,  -5.685792000e-03f,  4.363740000e-03f,  -4.505352000e-03f,  3.363444000e-03f, 
            -1.011004800e-02f,  9.037404000e-03f,  -8.654856000e-03f,  7.674228000e-03f,  -7.199664000e-03f,  6.311052000e-03f, 
            -5.744472000e-03f,  4.947876000e-03f,  -1.217342400e-02f,  1.171047600e-02f,  -1.044348000e-02f,  9.984420000e-03f, 
            -8.713536000e-03f,  8.258364000e-03f,  -6.983592000e-03f,  6.532308000e-03f,  -1.423680000e-02f,  1.438354800e-02f, 
            -1.223210400e-02f,  1.229461200e-02f,  -1.022740800e-02f,  1.020567600e-02f,  -8.222712000e-03f,  8.116740000e-03f, 
            -1.630017600e-02f,  1.705662000e-02f,  -1.402072800e-02f,  1.460480400e-02f,  -1.174128000e-02f,  1.215298800e-02f, 
            -9.461832000e-03f,  9.701172000e-03f,  -1.836355200e-02f,  1.972969200e-02f,  -1.580935200e-02f,  1.691499600e-02f, 
            -1.325515200e-02f,  1.410030000e-02f,  -1.070095200e-02f,  1.128560400e-02f,  -2.042692800e-02f,  2.240276400e-02f, 
            -1.759797600e-02f,  1.922518800e-02f,  -1.476902400e-02f,  1.604761200e-02f,  -1.194007200e-02f,  1.287003600e-02f, 
            -2.511035200e-02f,  2.234116000e-02f,  -2.122249600e-02f,  1.958024800e-02f,  -1.733464000e-02f,  1.681933600e-02f, 
            -1.344678400e-02f,  1.405842400e-02f,  -3.473340800e-02f,  1.730075200e-02f,  -2.924787200e-02f,  1.548606400e-02f, 
            -2.376233600e-02f,  1.367137600e-02f,  -1.827680000e-02f,  1.185668800e-02f,  -4.358104000e-02f,  2.310886400e-02f, 
            -3.793480000e-02f,  1.951318400e-02f,  -3.228856000e-02f,  1.591750400e-02f,  -2.664232000e-02f,  1.232182400e-02f, 
            -3.990041600e-02f,  3.147468000e-02f,  -3.453512000e-02f,  2.697655200e-02f,  -2.916982400e-02f,  2.247842400e-02f, 
            -2.380452800e-02f,  1.798029600e-02f,  -4.035580800e-02f,  3.566012400e-02f,  -3.485709600e-02f,  3.066814800e-02f, 
            -2.935838400e-02f,  2.567617200e-02f,  -2.385967200e-02f,  2.068419600e-02f,  -4.241918400e-02f,  3.833319600e-02f, 
            -3.664572000e-02f,  3.297834000e-02f,  -3.087225600e-02f,  2.762348400e-02f,  -2.509879200e-02f,  2.226862800e-02f, 
            -4.448256000e-02f,  4.100626800e-02f,  -3.843434400e-02f,  3.528853200e-02f,  -3.238612800e-02f,  2.957079600e-02f, 
            -2.633791200e-02f,  2.385306000e-02f,  -4.654593600e-02f,  4.367934000e-02f,  -4.022296800e-02f,  3.759872400e-02f, 
            -3.390000000e-02f,  3.151810800e-02f,  -2.757703200e-02f,  2.543749200e-02f,  -4.860931200e-02f,  4.635241200e-02f, 
            -4.201159200e-02f,  3.990891600e-02f,  -3.541387200e-02f,  3.346542000e-02f,  -2.881615200e-02f,  2.702192400e-02f, 
            -5.067268800e-02f,  4.902548400e-02f,  -4.380021600e-02f,  4.221910800e-02f,  -3.692774400e-02f,  3.541273200e-02f, 
            -3.005527200e-02f,  2.860635600e-02f,  -5.273606400e-02f,  5.169855600e-02f,  -4.558884000e-02f,  4.452930000e-02f, 
            -3.844161600e-02f,  3.736004400e-02f,  -3.129439200e-02f,  3.019078800e-02f,  -5.996057600e-02f,  4.901349600e-02f, 
            -5.099412800e-02f,  4.300855200e-02f,  -4.202768000e-02f,  3.700360800e-02f,  -3.306123200e-02f,  3.099866400e-02f, 
            -7.669332800e-02f,  3.680003200e-02f,  -6.491268800e-02f,  3.293248000e-02f,  -5.313204800e-02f,  2.906492800e-02f, 
            -4.135140800e-02f,  2.519737600e-02f,  -8.391894400e-02f,  4.463566400e-02f,  -7.319411200e-02f,  3.777060800e-02f, 
            -6.246928000e-02f,  3.090555200e-02f,  -5.174444800e-02f,  2.404049600e-02f,  -7.393963200e-02f,  5.916077600e-02f, 
            -6.410400000e-02f,  5.081036000e-02f,  -5.426836800e-02f,  4.245994400e-02f,  -4.443273600e-02f,  3.410952800e-02f, 
            -7.266494400e-02f,  6.495591600e-02f,  -6.284796000e-02f,  5.597226000e-02f,  -5.303097600e-02f,  4.698860400e-02f, 
            -4.321399200e-02f,  3.800494800e-02f,  -7.472832000e-02f,  6.762898800e-02f,  -6.463658400e-02f,  5.828245200e-02f, 
            -5.454484800e-02f,  4.893591600e-02f,  -4.445311200e-02f,  3.958938000e-02f,  -7.679169600e-02f,  7.030206000e-02f, 
            -6.642520800e-02f,  6.059264400e-02f,  -5.605872000e-02f,  5.088322800e-02f,  -4.569223200e-02f,  4.117381200e-02f, 
            -7.885507200e-02f,  7.297513200e-02f,  -6.821383200e-02f,  6.290283600e-02f,  -5.757259200e-02f,  5.283054000e-02f, 
            -4.693135200e-02f,  4.275824400e-02f,  -8.091844800e-02f,  7.564820400e-02f,  -7.000245600e-02f,  6.521302800e-02f, 
            -5.908646400e-02f,  5.477785200e-02f,  -4.817047200e-02f,  4.434267600e-02f,  -8.298182400e-02f,  7.832127600e-02f, 
            -7.179108000e-02f,  6.752322000e-02f,  -6.060033600e-02f,  5.672516400e-02f,  -4.940959200e-02f,  4.592710800e-02f, 
            -8.504520000e-02f,  8.099434800e-02f,  -7.357970400e-02f,  6.983341200e-02f,  -6.211420800e-02f,  5.867247600e-02f, 
            -5.064871200e-02f,  4.751154000e-02f,  -9.481080000e-02f,  7.568583200e-02f,  -8.076576000e-02f,  6.643685600e-02f, 
            -6.672072000e-02f,  5.718788000e-02f,  -5.267568000e-02f,  4.793890400e-02f,  -1.186532480e-01f,  5.629931200e-02f, 
            -1.005775040e-01f,  5.037889600e-02f,  -8.250176000e-02f,  4.445848000e-02f,  -6.442601600e-02f,  3.853806400e-02f, 

        };

        float[] gw_expect = new float[] {
            -6.099334080e-01f,  6.270906720e-01f,  -6.594160320e-01f,  5.946296160e-01f,  -7.088986560e-01f,  5.621685600e-01f, 
            -6.169274880e-01f,  6.342249120e-01f,  -6.670406400e-01f,  6.013944480e-01f,  -7.171537920e-01f,  5.685639840e-01f, 
            -6.239215680e-01f,  6.413591520e-01f,  -6.746652480e-01f,  6.081592800e-01f,  -7.254089280e-01f,  5.749594080e-01f, 
            -6.309156480e-01f,  6.484933920e-01f,  -6.822898560e-01f,  6.149241120e-01f,  -7.336640640e-01f,  5.813548320e-01f, 
            -6.668189800e-01f,  6.205134640e-01f,  -7.174389240e-01f,  5.869259840e-01f,  -7.680588680e-01f,  5.533385040e-01f, 
            -6.744551320e-01f,  6.277854400e-01f,  -7.257289320e-01f,  5.938051280e-01f,  -7.770027320e-01f,  5.598248160e-01f, 
            -6.820912840e-01f,  6.350574160e-01f,  -7.340189400e-01f,  6.006842720e-01f,  -7.859465960e-01f,  5.663111280e-01f, 
            -6.897274360e-01f,  6.423293920e-01f,  -7.423089480e-01f,  6.075634160e-01f,  -7.948904600e-01f,  5.727974400e-01f, 
            -7.672199040e-01f,  5.715920160e-01f,  -8.166810240e-01f,  5.391739680e-01f,  -8.661421440e-01f,  5.067559200e-01f, 
            -7.759930560e-01f,  5.785158240e-01f,  -8.261062080e-01f,  5.457068640e-01f,  -8.762193600e-01f,  5.128979040e-01f, 
            -7.847662080e-01f,  5.854396320e-01f,  -8.355313920e-01f,  5.522397600e-01f,  -8.862965760e-01f,  5.190398880e-01f, 
            -7.935393600e-01f,  5.923634400e-01f,  -8.449565760e-01f,  5.587726560e-01f,  -8.963737920e-01f,  5.251818720e-01f, 
        };
    }
}
