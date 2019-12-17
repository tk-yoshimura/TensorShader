using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class DeconvolutionTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 2;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels, outchannels, kwidth), wval);

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field x_expect = Deconvolution1D(y, w);
            Field err = Abs(x_expect - x_actual);
            OutputNode err_node = err.Value.Save();

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] err_actual = err_node.Tensor.State;
            float[] gy_actual = y.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        float[] gy_expect = new float[] {
            1.507737000e-02f, 1.411270700e-02f, 1.314804400e-02f, 1.218338100e-02f, 1.121871800e-02f,
            1.025405500e-02f, 9.289392000e-03f, 8.324729000e-03f, 7.360066000e-03f, 6.395403000e-03f,
            5.430740000e-03f, 3.883747000e-02f, 3.668225400e-02f, 3.452703800e-02f, 3.237182200e-02f,
            3.021660600e-02f, 2.806139000e-02f, 2.590617400e-02f, 2.375095800e-02f, 2.159574200e-02f,
            1.944052600e-02f, 1.728531000e-02f, 6.780816000e-02f, 6.430176900e-02f, 6.079537800e-02f,
            5.728898700e-02f, 5.378259600e-02f, 5.027620500e-02f, 4.676981400e-02f, 4.326342300e-02f,
            3.975703200e-02f, 3.625064100e-02f, 3.274425000e-02f, 9.677885000e-02f, 9.192128400e-02f,
            8.706371800e-02f, 8.220615200e-02f, 7.734858600e-02f, 7.249102000e-02f, 6.763345400e-02f,
            6.277588800e-02f, 5.791832200e-02f, 5.306075600e-02f, 4.820319000e-02f, 1.257495400e-01f,
            1.195407990e-01f, 1.133320580e-01f, 1.071233170e-01f, 1.009145760e-01f, 9.470583500e-02f,
            8.849709400e-02f, 8.228835300e-02f, 7.607961200e-02f, 6.987087100e-02f, 6.366213000e-02f,
            7.846636000e-02f, 7.821636200e-02f, 7.796636400e-02f, 7.771636600e-02f, 7.746636800e-02f,
            7.721637000e-02f, 7.696637200e-02f, 7.671637400e-02f, 7.646637600e-02f, 7.621637800e-02f,
            7.596638000e-02f, 1.229938500e-01f, 1.163029490e-01f, 1.096120480e-01f, 1.029211470e-01f,
            9.623024600e-02f, 8.953934500e-02f, 8.284844400e-02f, 7.615754300e-02f, 6.946664200e-02f,
            6.277574100e-02f, 5.608484000e-02f, 1.906116100e-01f, 1.813783440e-01f, 1.721450780e-01f,
            1.629118120e-01f, 1.536785460e-01f, 1.444452800e-01f, 1.352120140e-01f, 1.259787480e-01f,
            1.167454820e-01f, 1.075122160e-01f, 9.827895000e-02f, 2.195823000e-01f, 2.089978590e-01f,
            1.984134180e-01f, 1.878289770e-01f, 1.772445360e-01f, 1.666600950e-01f, 1.560756540e-01f,
            1.454912130e-01f, 1.349067720e-01f, 1.243223310e-01f, 1.137378900e-01f, 2.485529900e-01f,
            2.366173740e-01f, 2.246817580e-01f, 2.127461420e-01f, 2.008105260e-01f, 1.888749100e-01f,
            1.769392940e-01f, 1.650036780e-01f, 1.530680620e-01f, 1.411324460e-01f, 1.291968300e-01f,
            2.775236800e-01f, 2.642368890e-01f, 2.509500980e-01f, 2.376633070e-01f, 2.243765160e-01f,
            2.110897250e-01f, 1.978029340e-01f, 1.845161430e-01f, 1.712293520e-01f, 1.579425610e-01f,
            1.446557700e-01f, 1.589908600e-01f, 1.584930200e-01f, 1.579951800e-01f, 1.574973400e-01f,
            1.569995000e-01f, 1.565016600e-01f, 1.560038200e-01f, 1.555059800e-01f, 1.550081400e-01f,
            1.545103000e-01f, 1.540124600e-01f,
        };

        float[] gw_expect = new float[] {
            8.006031000e-02f, 7.803888400e-02f, 7.601745800e-02f, 7.399603200e-02f, 7.197460600e-02f,
            6.995318000e-02f, 6.793175400e-02f, 8.103394000e-02f, 7.898526800e-02f, 7.693659600e-02f,
            7.488792400e-02f, 7.283925200e-02f, 7.079058000e-02f, 6.874190800e-02f, 8.200757000e-02f,
            7.993165200e-02f, 7.785573400e-02f, 7.577981600e-02f, 7.370389800e-02f, 7.162798000e-02f,
            6.955206200e-02f, 8.298120000e-02f, 8.087803600e-02f, 7.877487200e-02f, 7.667170800e-02f,
            7.456854400e-02f, 7.246538000e-02f, 7.036221600e-02f, 8.395483000e-02f, 8.182442000e-02f,
            7.969401000e-02f, 7.756360000e-02f, 7.543319000e-02f, 7.330278000e-02f, 7.117237000e-02f,
            8.492846000e-02f, 8.277080400e-02f, 8.061314800e-02f, 7.845549200e-02f, 7.629783600e-02f,
            7.414018000e-02f, 7.198252400e-02f, 8.590209000e-02f, 8.371718800e-02f, 8.153228600e-02f,
            7.934738400e-02f, 7.716248200e-02f, 7.497758000e-02f, 7.279267800e-02f, 8.687572000e-02f,
            8.466357200e-02f, 8.245142400e-02f, 8.023927600e-02f, 7.802712800e-02f, 7.581498000e-02f,
            7.360283200e-02f, 8.784935000e-02f, 8.560995600e-02f, 8.337056200e-02f, 8.113116800e-02f,
            7.889177400e-02f, 7.665238000e-02f, 7.441298600e-02f, 8.882298000e-02f, 8.655634000e-02f,
            8.428970000e-02f, 8.202306000e-02f, 7.975642000e-02f, 7.748978000e-02f, 7.522314000e-02f,
            8.979661000e-02f, 8.750272400e-02f, 8.520883800e-02f, 8.291495200e-02f, 8.062106600e-02f,
            7.832718000e-02f, 7.603329400e-02f, -3.308932000e-03f, -4.748348000e-03f, -6.187764000e-03f,
            -7.627180000e-03f, -9.066596000e-03f, -1.050601200e-02f, -1.194542800e-02f, -3.347944000e-03f,
            -4.808006000e-03f, -6.268068000e-03f, -7.728130000e-03f, -9.188192000e-03f, -1.064825400e-02f,
            -1.210831600e-02f, -3.386956000e-03f, -4.867664000e-03f, -6.348372000e-03f, -7.829080000e-03f,
            -9.309788000e-03f, -1.079049600e-02f, -1.227120400e-02f, -3.425968000e-03f, -4.927322000e-03f,
            -6.428676000e-03f, -7.930030000e-03f, -9.431384000e-03f, -1.093273800e-02f, -1.243409200e-02f,
            -3.464980000e-03f, -4.986980000e-03f, -6.508980000e-03f, -8.030980000e-03f, -9.552980000e-03f,
            -1.107498000e-02f, -1.259698000e-02f, -3.503992000e-03f, -5.046638000e-03f, -6.589284000e-03f,
            -8.131930000e-03f, -9.674576000e-03f, -1.121722200e-02f, -1.275986800e-02f, -3.543004000e-03f,
            -5.106296000e-03f, -6.669588000e-03f, -8.232880000e-03f, -9.796172000e-03f, -1.135946400e-02f,
            -1.292275600e-02f, -3.582016000e-03f, -5.165954000e-03f, -6.749892000e-03f, -8.333830000e-03f,
            -9.917768000e-03f, -1.150170600e-02f, -1.308564400e-02f, -3.621028000e-03f, -5.225612000e-03f,
            -6.830196000e-03f, -8.434780000e-03f, -1.003936400e-02f, -1.164394800e-02f, -1.324853200e-02f,
            -3.660040000e-03f, -5.285270000e-03f, -6.910500000e-03f, -8.535730000e-03f, -1.016096000e-02f,
            -1.178619000e-02f, -1.341142000e-02f, -3.699052000e-03f, -5.344928000e-03f, -6.990804000e-03f,
            -8.636680000e-03f, -1.028255600e-02f, -1.192843200e-02f, -1.357430800e-02f, 4.856473600e-02f,
            4.654936000e-02f, 4.453398400e-02f, 4.251860800e-02f, 4.050323200e-02f, 3.848785600e-02f,
            3.647248000e-02f, 4.929123200e-02f, 4.724740000e-02f, 4.520356800e-02f, 4.315973600e-02f,
            4.111590400e-02f, 3.907207200e-02f, 3.702824000e-02f, 5.001772800e-02f, 4.794544000e-02f,
            4.587315200e-02f, 4.380086400e-02f, 4.172857600e-02f, 3.965628800e-02f, 3.758400000e-02f,
            5.074422400e-02f, 4.864348000e-02f, 4.654273600e-02f, 4.444199200e-02f, 4.234124800e-02f,
            4.024050400e-02f, 3.813976000e-02f, 5.147072000e-02f, 4.934152000e-02f, 4.721232000e-02f,
            4.508312000e-02f, 4.295392000e-02f, 4.082472000e-02f, 3.869552000e-02f, 5.219721600e-02f,
            5.003956000e-02f, 4.788190400e-02f, 4.572424800e-02f, 4.356659200e-02f, 4.140893600e-02f,
            3.925128000e-02f, 5.292371200e-02f, 5.073760000e-02f, 4.855148800e-02f, 4.636537600e-02f,
            4.417926400e-02f, 4.199315200e-02f, 3.980704000e-02f, 5.365020800e-02f, 5.143564000e-02f,
            4.922107200e-02f, 4.700650400e-02f, 4.479193600e-02f, 4.257736800e-02f, 4.036280000e-02f,
            5.437670400e-02f, 5.213368000e-02f, 4.989065600e-02f, 4.764763200e-02f, 4.540460800e-02f,
            4.316158400e-02f, 4.091856000e-02f, 5.510320000e-02f, 5.283172000e-02f, 5.056024000e-02f,
            4.828876000e-02f, 4.601728000e-02f, 4.374580000e-02f, 4.147432000e-02f, 5.582969600e-02f,
            5.352976000e-02f, 5.122982400e-02f, 4.892988800e-02f, 4.662995200e-02f, 4.433001600e-02f,
            4.203008000e-02f,
        };
    }
}
