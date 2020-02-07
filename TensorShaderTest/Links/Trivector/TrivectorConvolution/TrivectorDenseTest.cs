using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorConvolution {
    [TestClass]
    public class TrivectorDenseTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, batch = 3;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-2f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-2f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-2f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map0D(inchannels, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map0D(outchannels, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field y_expect = TrivectorDense(x, w);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-6f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int inchannels = 9, outchannels = 12, batch = 3;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-2f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-2f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-2f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map0D(inchannels, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map0D(outchannels, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field xx = TrivectorX(x), xy = TrivectorY(x), xz = TrivectorZ(x);
            Field wr = QuaternionR(w), wi = QuaternionI(w), wj = QuaternionJ(w), wk = QuaternionK(w);
            Field wrr = wr * wr, wii = wi * wi, wjj = wj * wj, wkk = wk * wk;
            Field wri = wr * wi, wrj = wr * wj, wrk = wr * wk, wij = wi * wj, wik = wi * wk, wjk = wj * wk;

            Field yx = Dense(xx, (wrr + wii - wjj - wkk)) + 2 * (Dense(xy, (wij - wrk)) + Dense(xz, (wik + wrj)));
            Field yy = Dense(xy, (wrr - wii + wjj - wkk)) + 2 * (Dense(xz, (wjk - wri)) + Dense(xx, (wij + wrk)));
            Field yz = Dense(xz, (wrr - wii - wjj + wkk)) + 2 * (Dense(xx, (wik - wrj)) + Dense(xy, (wjk + wri)));

            Field y_expect = TrivectorCast(yx, yy, yz);

            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            1.484675360e-02f,  2.119646720e-02f,  7.992881120e-02f,  1.685052960e-02f,  2.224692480e-02f,  6.862844640e-02f,
            1.709261600e-02f,  2.153067520e-02f,  5.715226080e-02f,  3.405597440e-02f,  3.984351200e-02f,  1.162607840e-01f,
            3.863938560e-02f,  4.346531040e-02f,  1.035381216e-01f,  3.928828160e-02f,  4.316416480e-02f,  8.886494400e-02f,
            5.326519520e-02f,  5.849055680e-02f,  1.525927568e-01f,  6.042824160e-02f,  6.468369600e-02f,  1.384477968e-01f,
            6.148394720e-02f,  6.479765440e-02f,  1.205776272e-01f
        };

        float[] gw_expect = new float[] {
            1.468720104e-01f,  1.674089856e-01f,  1.433720184e-01f,  1.155097704e-01f,  1.651917900e-01f,  1.913858856e-01f,
            1.583911428e-01f,  1.266252384e-01f,  1.778128848e-01f,  2.076212016e-01f,  1.680788976e-01f,  1.337176056e-01f,
            -2.702403120e-02f,  -1.161187200e-03f,  -2.581634640e-02f,  -4.594994160e-02f,  -2.860040040e-02f,  -4.285512000e-04f,
            -2.704006200e-02f,  -4.967607840e-02f,  -2.894340960e-02f,  2.451888000e-04f,  -2.696999520e-02f,  -5.093702160e-02f,
            -8.097060720e-02f,  -5.481521280e-02f,  -7.537319760e-02f,  -8.896423920e-02f,  -8.161381320e-02f,  -5.445088560e-02f,
            -7.367761560e-02f,  -8.857512960e-02f,  -7.603250400e-02f,  -4.860272160e-02f,  -6.590445600e-02f,  -8.107388880e-02f,
            -5.632359600e-02f,  -3.401317440e-02f,  -4.710231120e-02f,  -5.533689840e-02f,  -4.393833000e-02f,  -2.077099920e-02f,
            -3.161139960e-02f,  -4.016179680e-02f,  -2.227828320e-02f,  1.357790400e-03f,  -7.100472000e-03f,  -1.506898320e-02f,
        };
    }
}
