using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexConvolution {
    [TestClass]
    public class ComplexConvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            ParameterField x = (Shape.Map1D(inchannels, inwidth, batch), xval);
            ParameterField w = (Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);
            VariableField y_actual = (Shape.Map1D(outchannels, outwidth, batch), yval);

            Field y_expect = ComplexConvolution1D(x, w);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;
            float[] gw_actual = w.GradState.Value;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            ParameterField x = (Shape.Map1D(inchannels, inwidth, batch), xval);
            ParameterField w = (Shape.Kernel1D(inchannels, outchannels / 2, kwidth), wval);
            VariableField y_actual = (Shape.Map1D(outchannels, outwidth, batch), yval);

            Field x_real = ComplexReal(x), x_imag = ComplexImag(x);
            Field w_real = ComplexReal(w), w_imag = ComplexImag(w);

            Field y_real = Convolution1D(x_real, w_real) - Convolution1D(x_imag, w_imag);
            Field y_imag = Convolution1D(x_imag, w_real) + Convolution1D(x_real, w_imag);

            Field y_expect = ComplexCast(y_real, y_imag);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState.Value;
            float[] gw_actual = w.GradState.Value;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        readonly float[] gx_expect = new float[] {
            -7.413040000e-04f,  7.361320000e-04f,  -7.126400000e-04f,  7.129080000e-04f,  -6.839760000e-04f,  6.896840000e-04f,
            -4.106312000e-03f,  2.118672000e-03f,  -3.952088000e-03f,  2.041984000e-03f,  -3.797864000e-03f,  1.965296000e-03f,
            -8.932272000e-03f,  3.784740000e-03f,  -8.555592000e-03f,  3.624348000e-03f,  -8.178912000e-03f,  3.463956000e-03f,
            -1.434703200e-02f,  5.471388000e-03f,  -1.367966400e-02f,  5.220276000e-03f,  -1.301229600e-02f,  4.969164000e-03f,
            -1.976179200e-02f,  7.158036000e-03f,  -1.880373600e-02f,  6.816204000e-03f,  -1.784568000e-02f,  6.474372000e-03f,
            -2.517655200e-02f,  8.844684000e-03f,  -2.392780800e-02f,  8.412132000e-03f,  -2.267906400e-02f,  7.979580000e-03f,
            -3.059131200e-02f,  1.053133200e-02f,  -2.905188000e-02f,  1.000806000e-02f,  -2.751244800e-02f,  9.484788000e-03f,
            -3.600607200e-02f,  1.221798000e-02f,  -3.417595200e-02f,  1.160398800e-02f,  -3.234583200e-02f,  1.098999600e-02f,
            -4.142083200e-02f,  1.390462800e-02f,  -3.930002400e-02f,  1.319991600e-02f,  -3.717921600e-02f,  1.249520400e-02f,
            -4.683559200e-02f,  1.559127600e-02f,  -4.442409600e-02f,  1.479584400e-02f,  -4.201260000e-02f,  1.400041200e-02f,
            -5.225035200e-02f,  1.727792400e-02f,  -4.954816800e-02f,  1.639177200e-02f,  -4.684598400e-02f,  1.550562000e-02f,
            -2.427941600e-02f,  8.052384000e-03f,  -2.238106400e-02f,  7.431376000e-03f,  -2.048271200e-02f,  6.810368000e-03f,
            -6.475048000e-03f,  2.172116000e-03f,  -5.477424000e-03f,  1.846492000e-03f,  -4.479800000e-03f,  1.520868000e-03f,
            -3.144904000e-02f,  1.282638000e-02f,  -3.041672800e-02f,  1.241003600e-02f,  -2.938441600e-02f,  1.199369200e-02f,
            -5.347800800e-02f,  2.158172800e-02f,  -5.131648800e-02f,  2.071880000e-02f,  -4.915496800e-02f,  1.985587200e-02f,
            -6.492415200e-02f,  2.590316400e-02f,  -6.153652800e-02f,  2.456341200e-02f,  -5.814890400e-02f,  2.322366000e-02f,
            -7.033891200e-02f,  2.758981200e-02f,  -6.666060000e-02f,  2.615934000e-02f,  -6.298228800e-02f,  2.472886800e-02f,
            -7.575367200e-02f,  2.927646000e-02f,  -7.178467200e-02f,  2.775526800e-02f,  -6.781567200e-02f,  2.623407600e-02f,
            -8.116843200e-02f,  3.096310800e-02f,  -7.690874400e-02f,  2.935119600e-02f,  -7.264905600e-02f,  2.773928400e-02f,
            -8.658319200e-02f,  3.264975600e-02f,  -8.203281600e-02f,  3.094712400e-02f,  -7.748244000e-02f,  2.924449200e-02f,
            -9.199795200e-02f,  3.433640400e-02f,  -8.715688800e-02f,  3.254305200e-02f,  -8.231582400e-02f,  3.074970000e-02f,
            -9.741271200e-02f,  3.602305200e-02f,  -9.228096000e-02f,  3.413898000e-02f,  -8.714920800e-02f,  3.225490800e-02f,
            -1.028274720e-01f,  3.770970000e-02f,  -9.740503200e-02f,  3.573490800e-02f,  -9.198259200e-02f,  3.376011600e-02f,
            -1.082422320e-01f,  3.939634800e-02f,  -1.025291040e-01f,  3.733083600e-02f,  -9.681597600e-02f,  3.526532400e-02f,
            -4.956356000e-02f,  1.808056000e-02f,  -4.565791200e-02f,  1.667331200e-02f,  -4.175226400e-02f,  1.526606400e-02f,
            -1.309523200e-02f,  4.827484000e-03f,  -1.109396000e-02f,  4.108740000e-03f,  -9.092688000e-03f,  3.389996000e-03f,
            -6.215677600e-02f,  2.491662800e-02f,  -6.012081600e-02f,  2.410716400e-02f,  -5.808485600e-02f,  2.329770000e-02f,
            -1.028497040e-01f,  4.104478400e-02f,  -9.868088800e-02f,  3.939561600e-02f,  -9.451207200e-02f,  3.774644800e-02f,
            -1.209160320e-01f,  4.802158800e-02f,  -1.145174640e-01f,  4.550247600e-02f,  -1.081188960e-01f,  4.298336400e-02f,
            -1.263307920e-01f,  4.970823600e-02f,  -1.196415360e-01f,  4.709840400e-02f,  -1.129522800e-01f,  4.448857200e-02f,
            -1.317455520e-01f,  5.139488400e-02f,  -1.247656080e-01f,  4.869433200e-02f,  -1.177856640e-01f,  4.599378000e-02f,
            -1.371603120e-01f,  5.308153200e-02f,  -1.298896800e-01f,  5.029026000e-02f,  -1.226190480e-01f,  4.749898800e-02f,
            -1.425750720e-01f,  5.476818000e-02f,  -1.350137520e-01f,  5.188618800e-02f,  -1.274524320e-01f,  4.900419600e-02f,
            -1.479898320e-01f,  5.645482800e-02f,  -1.401378240e-01f,  5.348211600e-02f,  -1.322858160e-01f,  5.050940400e-02f,
            -1.534045920e-01f,  5.814147600e-02f,  -1.452618960e-01f,  5.507804400e-02f,  -1.371192000e-01f,  5.201461200e-02f,
            -1.588193520e-01f,  5.982812400e-02f,  -1.503859680e-01f,  5.667397200e-02f,  -1.419525840e-01f,  5.351982000e-02f,
            -1.642341120e-01f,  6.151477200e-02f,  -1.555100400e-01f,  5.826990000e-02f,  -1.467859680e-01f,  5.502502800e-02f,
            -7.484770400e-02f,  2.810873600e-02f,  -6.893476000e-02f,  2.591524800e-02f,  -6.302181600e-02f,  2.372176000e-02f,
            -1.971541600e-02f,  7.482852000e-03f,  -1.671049600e-02f,  6.370988000e-03f,  -1.370557600e-02f,  5.259124000e-03f,
        };
        readonly float[] gw_expect = new float[] {
            -8.042644830e-01f,  4.400121000e-01f,  -8.151417450e-01f,  4.459453020e-01f,  -8.260190070e-01f,  4.518785040e-01f,
            -8.793901050e-01f,  3.793029900e-01f,  -8.913582150e-01f,  3.844022160e-01f,  -9.033263250e-01f,  3.895014420e-01f,
            -9.545157270e-01f,  3.185938800e-01f,  -9.675746850e-01f,  3.228591300e-01f,  -9.806336430e-01f,  3.271243800e-01f,
            -1.029641349e+00f,  2.578847700e-01f,  -1.043791155e+00f,  2.613160440e-01f,  -1.057940961e+00f,  2.647473180e-01f,
            -8.368962690e-01f,  4.578117060e-01f,  -8.477735310e-01f,  4.637449080e-01f,  -8.586507930e-01f,  4.696781100e-01f,
            -9.152944350e-01f,  3.946006680e-01f,  -9.272625450e-01f,  3.996998940e-01f,  -9.392306550e-01f,  4.047991200e-01f,
            -9.936926010e-01f,  3.313896300e-01f,  -1.006751559e+00f,  3.356548800e-01f,  -1.019810517e+00f,  3.399201300e-01f,
            -1.072090767e+00f,  2.681785920e-01f,  -1.086240573e+00f,  2.716098660e-01f,  -1.100390379e+00f,  2.750411400e-01f,
            -8.695280550e-01f,  4.756113120e-01f,  -8.804053170e-01f,  4.815445140e-01f,  -8.912825790e-01f,  4.874777160e-01f,
            -9.511987650e-01f,  4.098983460e-01f,  -9.631668750e-01f,  4.149975720e-01f,  -9.751349850e-01f,  4.200967980e-01f,
            -1.032869475e+00f,  3.441853800e-01f,  -1.045928433e+00f,  3.484506300e-01f,  -1.058987391e+00f,  3.527158800e-01f,
            -1.114540185e+00f,  2.784724140e-01f,  -1.128689991e+00f,  2.819036880e-01f,  -1.142839797e+00f,  2.853349620e-01f,
        };
    }
}
