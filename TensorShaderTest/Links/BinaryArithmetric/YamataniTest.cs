using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.BinaryArithmetric {
    [TestClass]
    public class YamataniTest {
        const float slope = 0.1f;

        [TestMethod]
        public void ReferenceTest() {
            int length = 128;

            float[] x1val = (new float[length]).Select((_, idx) => (float)Math.Sin(idx * 0.1) * idx * 0.1f).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)Math.Cos(idx * 0.1) * idx * 0.1f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x1 = x1val;
            ParameterField x2 = x2val;
            VariableField y_actual = yval;

            Field y_expect = Yamatani(x1, x2, slope);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState.Value;
            float[] gx2_actual = x2.GradState.Value;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        [TestMethod]
        public void TheoreticalTest() {
            int length = 128;

            float[] x1val = (new float[length]).Select((_, idx) => (float)Math.Sin(idx * 0.1) * idx * 0.1f).ToArray();
            float[] x2val = (new float[length]).Select((_, idx) => (float)Math.Cos(idx * 0.1) * idx * 0.1f).ToArray();
            float[] yval = (new float[length]).Select((_, idx) => (float)idx / 24).ToArray();

            ParameterField x1 = x1val;
            ParameterField x2 = x2val;
            VariableField y_actual = yval;

            Field s = slope * (x1 + x2);
            Field pp = Minimum(x1, x2) * GreaterThan(x1, 0) * GreaterThan(x2, 0);
            Field nn = Maximum(x1, x2) * LessThan(x1, 0) * LessThan(x2, 0);

            StoreField y_expect = s + pp + nn;
            Field err = y_expect - y_actual;

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState.Value;
            float[] gx2_actual = x2.GradState.Value;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");
        }

        readonly float[] gx1_expect = {
            0.00000000e+00f,  -2.28084441e-02f,  -2.20272239e-02f,  1.29993916e-03f,  4.56718281e-02f,  1.09152825e-01f,  1.89402586e-01f,  2.83711896e-01f,
            3.37157058e-02f,  3.10893290e-02f,  2.61812968e-02f,  1.88550783e-02f,  9.01569262e-03f,  -3.38807776e-03f,  -1.83620971e-02f,  -3.58659369e-02f,
            -5.11406814e-02f,  -5.61653880e-02f,  -6.15603803e-02f,  -6.73294668e-02f,  -7.34703215e-02f,  -7.99743715e-02f,  -8.68267704e-02f,  -9.40064619e-02f,
            -1.01486333e-01f,  -1.09233453e-01f,  -1.17209405e-01f,  -1.25370691e-01f,  -1.33669224e-01f,  -1.42052890e-01f,  -1.50466175e-01f,  -1.58850856e-01f,
            -2.04409111e+00f,  -2.50083371e+00f,  -2.97121195e+00f,  -3.45026946e+00f,  -3.93273538e+00f,  -4.41308645e+00f,  -4.88561375e+00f,  -5.34449346e+00f,
            -4.84541960e-01f,  -4.63628295e-01f,  -4.38106682e-01f,  -4.08139810e-01f,  -3.73952934e-01f,  -3.35832776e-01f,  -2.94125631e-01f,  -2.49234665e-01f,
            -2.43615950e-01f,  -2.43167739e-01f,  -2.42096438e-01f,  -2.40439684e-01f,  -2.38243442e-01f,  -2.35561668e-01f,  -2.32455867e-01f,  -2.28994547e-01f,
            -2.25252576e-01f,  -2.21310447e-01f,  -2.17253461e-01f,  -2.13170829e-01f,  -2.09154713e-01f,  -2.05299205e-01f,  -2.01699266e-01f,  -2.06642560e+00f,
            -1.33157411e+00f,  -5.88987808e-01f,  1.52838191e-01f,  8.85185353e-01f,  1.59920731e+00f,  2.28602894e+00f,  2.93684789e+00f,  2.90554415e-01f,
            2.38958327e-01f,  1.80355453e-01f,  1.15148590e-01f,  4.38241355e-02f,  -3.30515432e-02f,  -1.14836718e-01f,  -2.00819868e-01f,  -2.53884469e-01f,
            -2.65824676e-01f,  -2.78666002e-01f,  -2.92337460e-01f,  -3.06757349e-01f,  -3.21833939e-01f,  -3.37466274e-01f,  -3.53545107e-01f,  -3.69953935e-01f,
            -3.86570139e-01f,  -4.03266211e-01f,  -4.19911060e-01f,  -4.36371381e-01f,  -4.52513074e-01f,  -4.68202694e-01f,  -4.83308923e-01f,  -6.26007370e+00f,
            -7.46481027e+00f,  -8.66233113e+00f,  -9.84037570e+00f,  -1.09866516e+01f,  -1.20889675e+01f,  -1.31353663e+01f,  -1.41142595e+01f,  -1.23429464e+00f,
            -1.16119328e+00f,  -1.07911321e+00f,  -9.88785192e-01f,  -8.91040765e-01f,  -7.86805042e-01f,  -6.77088411e-01f,  -5.67845429e-01f,  -5.61325138e-01f,
            -5.53598028e-01f,  -5.44766285e-01f,  -5.34945493e-01f,  -5.24263619e-01f,  -5.12859853e-01f,  -5.00883308e-01f,  -4.88491594e-01f,  -4.75849284e-01f,
            -4.63126275e-01f,  -4.50496070e-01f,  -4.38134001e-01f,  -4.26215398e-01f,  -4.14913739e-01f,  -4.04398786e-01f,  -3.87716669e+00f,  -2.38891099e+00f,
        };
        readonly float[] gx2_expect = {
            0.00000000e+00f,  -2.07349492e-03f,  -2.00247490e-03f,  1.18176287e-04f,  4.15198437e-03f,  9.92298410e-03f,  1.72184169e-02f,  2.57919906e-02f,
            3.70872764e-01f,  3.41982619e-01f,  2.87994265e-01f,  2.07405862e-01f,  9.91726188e-02f,  -3.72688554e-02f,  -2.01983068e-01f,  -3.94525306e-01f,
            -5.11406814e-02f,  -5.61653880e-02f,  -6.15603803e-02f,  -6.73294668e-02f,  -7.34703215e-02f,  -7.99743715e-02f,  -8.68267704e-02f,  -9.40064619e-02f,
            -1.01486333e-01f,  -1.09233453e-01f,  -1.17209405e-01f,  -1.25370691e-01f,  -1.33669224e-01f,  -1.42052890e-01f,  -1.50466175e-01f,  -1.58850856e-01f,
            -1.85826465e-01f,  -2.27348519e-01f,  -2.70110177e-01f,  -3.13660860e-01f,  -3.57521399e-01f,  -4.01189677e-01f,  -4.44146705e-01f,  -4.85863042e-01f,
            -5.32996156e+00f,  -5.09991124e+00f,  -4.81917350e+00f,  -4.48953791e+00f,  -4.11348227e+00f,  -3.69416054e+00f,  -3.23538195e+00f,  -2.74158132e+00f,
            -2.43615950e-01f,  -2.43167739e-01f,  -2.42096438e-01f,  -2.40439684e-01f,  -2.38243442e-01f,  -2.35561668e-01f,  -2.32455867e-01f,  -2.28994547e-01f,
            -2.25252576e-01f,  -2.21310447e-01f,  -2.17253461e-01f,  -2.13170829e-01f,  -2.09154713e-01f,  -2.05299205e-01f,  -2.01699266e-01f,  -1.87856873e-01f,
            -1.21052192e-01f,  -5.35443462e-02f,  1.38943810e-02f,  8.04713958e-02f,  1.45382483e-01f,  2.07820813e-01f,  2.66986172e-01f,  3.19609857e+00f,
            2.62854160e+00f,  1.98390998e+00f,  1.26663449e+00f,  4.82065490e-01f,  -3.63566975e-01f,  -1.26320390e+00f,  -2.20901855e+00f,  -2.53884469e-01f,
            -2.65824676e-01f,  -2.78666002e-01f,  -2.92337460e-01f,  -3.06757349e-01f,  -3.21833939e-01f,  -3.37466274e-01f,  -3.53545107e-01f,  -3.69953935e-01f,
            -3.86570139e-01f,  -4.03266211e-01f,  -4.19911060e-01f,  -4.36371381e-01f,  -4.52513074e-01f,  -4.68202694e-01f,  -4.83308923e-01f,  -5.69097609e-01f,
            -6.78619115e-01f,  -7.87484648e-01f,  -8.94579609e-01f,  -9.98786512e-01f,  -1.09899704e+00f,  -1.19412421e+00f,  -1.28311450e+00f,  -1.35772410e+01f,
            -1.27731260e+01f,  -1.18702453e+01f,  -1.08766371e+01f,  -9.80144841e+00f,  -8.65485546e+00f,  -7.44797252e+00f,  -5.67845429e-01f,  -5.61325138e-01f,
            -5.53598028e-01f,  -5.44766285e-01f,  -5.34945493e-01f,  -5.24263619e-01f,  -5.12859853e-01f,  -5.00883308e-01f,  -4.88491594e-01f,  -4.75849284e-01f,
            -4.63126275e-01f,  -4.50496070e-01f,  -4.38134001e-01f,  -4.26215398e-01f,  -4.14913739e-01f,  -4.04398786e-01f,  -3.52469699e-01f,  -2.17173726e-01f,
        };
    }
}
