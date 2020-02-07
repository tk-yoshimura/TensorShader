using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class ChannelwiseDeconvolutionTest {
        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, kwidth = 3, inwidth = 13;
            int outwidth = inwidth - kwidth + 1, batch = 2;

            float[] xval = (new float[inwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * channels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(channels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(channels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(channels, 1, kwidth), wval);

            VariableField x_actual = xtensor;
            ParameterField w = wtensor;
            ParameterField y = ytensor;

            Field x_expect = ChannelwiseDeconvolution1D(y, w);
            StoreField err = Abs(x_expect - x_actual);
            
            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] err_actual = err.State;
            float[] gy_actual = y.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        float[] gy_expect = new float[] {
            -1.7095400e-04f,  -1.8574100e-04f,  -1.9474800e-04f,  -1.9793900e-04f,  -1.9527800e-04f,  -1.8672900e-04f,  -1.7225600e-04f,
            -4.3651300e-04f,  -4.3106200e-04f,  -4.1983900e-04f,  -4.0279600e-04f,  -3.7988500e-04f,  -3.5105800e-04f,  -3.1626700e-04f,
            -6.9970600e-04f,  -6.7456000e-04f,  -6.4357600e-04f,  -6.0670000e-04f,  -5.6387800e-04f,  -5.1505600e-04f,  -4.6018000e-04f,
            -9.6205900e-04f,  -9.1748800e-04f,  -8.6695300e-04f,  -8.1040000e-04f,  -7.4777500e-04f,  -6.7902400e-04f,  -6.0409300e-04f,
            -1.2244120e-03f,  -1.1604160e-03f,  -1.0903300e-03f,  -1.0141000e-03f,  -9.3167200e-04f,  -8.4299200e-04f,  -7.4800600e-04f,
            -1.4867650e-03f,  -1.4033440e-03f,  -1.3137070e-03f,  -1.2178000e-03f,  -1.1155690e-03f,  -1.0069600e-03f,  -8.9191900e-04f,
            -1.7491180e-03f,  -1.6462720e-03f,  -1.5370840e-03f,  -1.4215000e-03f,  -1.2994660e-03f,  -1.1709280e-03f,  -1.0358320e-03f,
            -2.0114710e-03f,  -1.8892000e-03f,  -1.7604610e-03f,  -1.6252000e-03f,  -1.4833630e-03f,  -1.3348960e-03f,  -1.1797450e-03f,
            -2.2738240e-03f,  -2.1321280e-03f,  -1.9838380e-03f,  -1.8289000e-03f,  -1.6672600e-03f,  -1.4988640e-03f,  -1.3236580e-03f,
            -2.5454170e-03f,  -2.3824660e-03f,  -2.2129030e-03f,  -2.0366800e-03f,  -1.8537490e-03f,  -1.6640620e-03f,  -1.4675710e-03f,
            -2.8346360e-03f,  -2.6485230e-03f,  -2.4559020e-03f,  -2.2567370e-03f,  -2.0509920e-03f,  -1.8386310e-03f,  -1.6196180e-03f,
            -3.6381030e-03f,  -3.3914400e-03f,  -3.1380730e-03f,  -2.8779660e-03f,  -2.6110830e-03f,  -2.3373880e-03f,  -2.0568450e-03f,
            -3.8776360e-03f,  -3.6145850e-03f,  -3.3445300e-03f,  -3.0674230e-03f,  -2.7832160e-03f,  -2.4918610e-03f,  -2.1933100e-03f,
            -4.1315890e-03f,  -3.8507680e-03f,  -3.5627230e-03f,  -3.2674000e-03f,  -2.9647450e-03f,  -2.6547040e-03f,  -2.3372230e-03f,
            -4.3939420e-03f,  -4.0936960e-03f,  -3.7861000e-03f,  -3.4711000e-03f,  -3.1486420e-03f,  -2.8186720e-03f,  -2.4811360e-03f,
            -4.6562950e-03f,  -4.3366240e-03f,  -4.0094770e-03f,  -3.6748000e-03f,  -3.3325390e-03f,  -2.9826400e-03f,  -2.6250490e-03f,
            -4.9186480e-03f,  -4.5795520e-03f,  -4.2328540e-03f,  -3.8785000e-03f,  -3.5164360e-03f,  -3.1466080e-03f,  -2.7689620e-03f,
            -5.1810010e-03f,  -4.8224800e-03f,  -4.4562310e-03f,  -4.0822000e-03f,  -3.7003330e-03f,  -3.3105760e-03f,  -2.9128750e-03f,
            -5.4433540e-03f,  -5.0654080e-03f,  -4.6796080e-03f,  -4.2859000e-03f,  -3.8842300e-03f,  -3.4745440e-03f,  -3.0567880e-03f,
            -5.7057070e-03f,  -5.3083360e-03f,  -4.9029850e-03f,  -4.4896000e-03f,  -4.0681270e-03f,  -3.6385120e-03f,  -3.2007010e-03f,
            -5.9865400e-03f,  -5.5659890e-03f,  -5.1375940e-03f,  -4.7013070e-03f,  -4.2570800e-03f,  -3.8048650e-03f,  -3.3446140e-03f,
            -6.3017850e-03f,  -5.8542220e-03f,  -5.3992270e-03f,  -4.9367640e-03f,  -4.4667970e-03f,  -3.9892900e-03f,  -3.5042070e-03f,
        };

        float[] gw_expect = new float[] {
            -1.7357682e-01f,  -1.7731280e-01f,  -1.8110917e-01f,  -1.8496628e-01f,  -1.8888450e-01f,  -1.9286418e-01f,  -1.9690568e-01f,
            -1.8491311e-01f,  -1.8882030e-01f,  -1.9278847e-01f,  -1.9681801e-01f,  -2.0090928e-01f,  -2.0506265e-01f,  -2.0927849e-01f,
            -1.9668130e-01f,  -2.0073357e-01f,  -2.0484634e-01f,  -2.0901996e-01f,  -2.1325480e-01f,  -2.1755122e-01f,  -2.2190957e-01f,
        };
    }
}
