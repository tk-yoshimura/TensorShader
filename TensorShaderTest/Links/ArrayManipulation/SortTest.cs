using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using TensorShader;
using TensorShader.Functions.ArrayManipulation;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ArrayManipulation {
    [TestClass]
    public class SortTest {
        [TestMethod]
        public void ReferenceWithBackpropTest() {
            Shape shape = new(ShapeType.Map, 5, 8, 7, 2);
            int length = shape.Length;

            float[] xval = (new float[length]).Select((_, idx) => (float)(idx * 0.0001f + ((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();

            float[] tval = (new float[length]).Select((_, idx) => (float)(idx * 2)).ToArray();

            ParameterField x = (shape, xval);
            VariableField t = (shape, tval);

            Field y = Sort(x, axis: 0);

            StoreField o = y.Save();

            Field err = y - t;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            Assert.IsTrue(flow.Nodes.Any((node) => node is FunctionNode funcnode && funcnode.Function is SortWithKey));

            flow.Execute();

            float[] y_actual = o.State;

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"not equal x");

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");

            flow.Execute();

            y_actual = o.State;

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"not equal x");

            gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        [TestMethod]
        public void ReferenceWithoutBackpropTest() {
            Shape shape = new(ShapeType.Map, 5, 8, 7, 2);
            int length = shape.Length;

            float[] xval = (new float[length]).Select((_, idx) => (float)(idx * 0.0001f + ((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();

            float[] tval = (new float[length]).Select((_, idx) => (float)(idx * 2)).ToArray();

            VariableField x = (shape, xval);
            VariableField t = (shape, tval);

            Field y = Sort(x, axis: 0);

            StoreField o = y.Save();

            Field err = y - t;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            Assert.IsFalse(flow.Nodes.Any((node) => node is FunctionNode funcnode && funcnode.Function is SortWithKey));

            flow.Execute();

            float[] y_actual = o.State;

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"not equal x");

            Assert.IsTrue(x.Grad is null);

            flow.Execute();

            y_actual = o.State;

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"not equal x");

            Assert.IsTrue(x.Grad is null);
        }

        readonly float[] y_expect = {
            4.0000e-04f,  1.0001e+00f,  1.0003e+00f,  2.0002e+00f,  7.0000e+00f,  2.0008e+00f,  3.0007e+00f,  4.0005e+00f,  6.0006e+00f,  7.0009e+00f,
            2.0010e+00f,  2.0013e+00f,  3.0014e+00f,  4.0011e+00f,  4.0012e+00f,  1.0016e+00f,  2.0018e+00f,  3.0017e+00f,  7.0015e+00f,  7.0019e+00f,
            1.0021e+00f,  1.0022e+00f,  2.0020e+00f,  4.0024e+00f,  7.0023e+00f,  2.6000e-03f,  1.0029e+00f,  4.0028e+00f,  6.0027e+00f,  7.0025e+00f,
            3.1000e-03f,  3.2000e-03f,  3.4000e-03f,  4.0030e+00f,  6.0033e+00f,  2.0038e+00f,  3.0035e+00f,  3.0037e+00f,  4.0036e+00f,  6.0039e+00f,
            4.0000e-03f,  1.0041e+00f,  1.0042e+00f,  3.0044e+00f,  7.0043e+00f,  1.0045e+00f,  1.0047e+00f,  1.0049e+00f,  2.0046e+00f,  5.0048e+00f,
            2.0053e+00f,  3.0050e+00f,  4.0052e+00f,  5.0051e+00f,  6.0054e+00f,  1.0055e+00f,  1.0057e+00f,  2.0056e+00f,  2.0058e+00f,  6.0059e+00f,
            6.0000e-03f,  6.1000e-03f,  3.0063e+00f,  6.0062e+00f,  6.0064e+00f,  1.0068e+00f,  2.0065e+00f,  3.0067e+00f,  4.0066e+00f,  5.0069e+00f,
            2.0072e+00f,  2.0074e+00f,  7.0070e+00f,  7.0071e+00f,  7.0073e+00f,  1.0077e+00f,  3.0075e+00f,  3.0076e+00f,  5.0078e+00f,  6.0079e+00f,
            8.0000e-03f,  8.1000e-03f,  8.2000e-03f,  8.4000e-03f,  5.0083e+00f,  8.8000e-03f,  3.0089e+00f,  4.0087e+00f,  6.0085e+00f,  6.0086e+00f,
            9.2000e-03f,  9.4000e-03f,  3.0091e+00f,  4.0090e+00f,  5.0093e+00f,  1.0095e+00f,  1.0097e+00f,  3.0096e+00f,  5.0098e+00f,  5.0099e+00f,
            1.0104e+00f,  3.0102e+00f,  5.0101e+00f,  6.0100e+00f,  7.0103e+00f,  1.0700e-02f,  1.0109e+00f,  2.0106e+00f,  3.0105e+00f,  4.0108e+00f,
            1.1300e-02f,  1.0111e+00f,  2.0110e+00f,  2.0114e+00f,  6.0112e+00f,  2.0115e+00f,  2.0116e+00f,  5.0118e+00f,  7.0117e+00f,  7.0119e+00f,
            1.2000e-02f,  3.0123e+00f,  4.0122e+00f,  5.0124e+00f,  7.0121e+00f,  2.0126e+00f,  2.0128e+00f,  3.0129e+00f,  5.0125e+00f,  7.0127e+00f,
            1.3300e-02f,  2.0134e+00f,  4.0132e+00f,  5.0130e+00f,  7.0131e+00f,  1.0137e+00f,  2.0135e+00f,  4.0136e+00f,  4.0138e+00f,  5.0139e+00f,
            1.0144e+00f,  2.0141e+00f,  3.0143e+00f,  4.0140e+00f,  6.0142e+00f,  1.4800e-02f,  1.0145e+00f,  1.0149e+00f,  5.0147e+00f,  7.0146e+00f,
            1.0153e+00f,  2.0154e+00f,  3.0151e+00f,  5.0150e+00f,  7.0152e+00f,  1.0155e+00f,  3.0157e+00f,  6.0156e+00f,  6.0158e+00f,  7.0159e+00f,
            1.6000e-02f,  2.0162e+00f,  2.0164e+00f,  4.0163e+00f,  6.0161e+00f,  1.0165e+00f,  2.0167e+00f,  4.0168e+00f,  4.0169e+00f,  6.0166e+00f,
            1.0173e+00f,  3.0171e+00f,  4.0174e+00f,  5.0170e+00f,  7.0172e+00f,  1.7600e-02f,  1.0177e+00f,  3.0175e+00f,  3.0178e+00f,  3.0179e+00f,
            1.8300e-02f,  1.8400e-02f,  1.0180e+00f,  2.0182e+00f,  7.0181e+00f,  2.0187e+00f,  4.0186e+00f,  4.0188e+00f,  4.0189e+00f,  7.0185e+00f,
            1.9200e-02f,  1.0193e+00f,  2.0190e+00f,  2.0194e+00f,  5.0191e+00f,  1.9500e-02f,  2.0196e+00f,  5.0197e+00f,  7.0198e+00f,  7.0199e+00f,
            2.0200e-02f,  1.0203e+00f,  5.0200e+00f,  5.0201e+00f,  7.0204e+00f,  1.0206e+00f,  3.0207e+00f,  4.0209e+00f,  5.0205e+00f,  6.0208e+00f,
            2.0210e+00f,  2.0212e+00f,  2.0214e+00f,  3.0213e+00f,  7.0211e+00f,  2.1600e-02f,  1.0219e+00f,  2.0217e+00f,  2.0218e+00f,  4.0215e+00f,
            4.0221e+00f,  6.0220e+00f,  6.0222e+00f,  6.0224e+00f,  7.0223e+00f,  2.2800e-02f,  1.0226e+00f,  3.0227e+00f,  5.0225e+00f,  7.0229e+00f,
            1.0232e+00f,  1.0233e+00f,  4.0230e+00f,  7.0231e+00f,  7.0234e+00f,  2.3500e-02f,  2.3800e-02f,  4.0236e+00f,  6.0237e+00f,  7.0239e+00f,
            4.0240e+00f,  4.0244e+00f,  6.0242e+00f,  6.0243e+00f,  7.0241e+00f,  1.0245e+00f,  1.0249e+00f,  3.0248e+00f,  4.0246e+00f,  5.0247e+00f,
            1.0251e+00f,  3.0254e+00f,  5.0252e+00f,  5.0253e+00f,  6.0250e+00f,  2.5600e-02f,  2.5800e-02f,  1.0257e+00f,  5.0255e+00f,  7.0259e+00f,
            3.0260e+00f,  4.0264e+00f,  5.0261e+00f,  6.0262e+00f,  6.0263e+00f,  2.6500e-02f,  2.0269e+00f,  6.0266e+00f,  6.0268e+00f,  7.0267e+00f,
            2.7100e-02f,  1.0273e+00f,  2.0272e+00f,  6.0270e+00f,  7.0274e+00f,  3.0275e+00f,  6.0276e+00f,  6.0278e+00f,  7.0277e+00f,  7.0279e+00f,
            2.8400e-02f,  3.0280e+00f,  3.0283e+00f,  5.0281e+00f,  5.0282e+00f,  2.8800e-02f,  1.0287e+00f,  3.0285e+00f,  6.0289e+00f,  7.0286e+00f,
            2.0290e+00f,  4.0291e+00f,  4.0292e+00f,  4.0294e+00f,  7.0293e+00f,  2.9600e-02f,  2.9700e-02f,  2.0299e+00f,  6.0298e+00f,  7.0295e+00f,
            2.0304e+00f,  3.0300e+00f,  5.0301e+00f,  5.0302e+00f,  5.0303e+00f,  1.0308e+00f,  3.0306e+00f,  3.0307e+00f,  5.0305e+00f,  5.0309e+00f,
            3.1000e-02f,  3.1200e-02f,  1.0311e+00f,  2.0314e+00f,  5.0313e+00f,  3.1700e-02f,  2.0319e+00f,  5.0315e+00f,  6.0316e+00f,  6.0318e+00f,
            3.2300e-02f,  2.0320e+00f,  2.0322e+00f,  3.0321e+00f,  4.0324e+00f,  3.2600e-02f,  1.0329e+00f,  5.0328e+00f,  6.0325e+00f,  7.0327e+00f,
            3.0333e+00f,  5.0334e+00f,  6.0330e+00f,  6.0332e+00f,  7.0331e+00f,  2.0339e+00f,  4.0338e+00f,  7.0335e+00f,  7.0336e+00f,  7.0337e+00f,
            3.4400e-02f,  1.0343e+00f,  4.0340e+00f,  4.0342e+00f,  5.0341e+00f,  2.0345e+00f,  2.0346e+00f,  4.0348e+00f,  7.0347e+00f,  7.0349e+00f,
            3.5000e-02f,  1.0353e+00f,  2.0351e+00f,  4.0352e+00f,  5.0354e+00f,  1.0357e+00f,  1.0359e+00f,  6.0358e+00f,  7.0355e+00f,  7.0356e+00f,
            3.6400e-02f,  1.0361e+00f,  2.0360e+00f,  3.0363e+00f,  7.0362e+00f,  2.0368e+00f,  4.0367e+00f,  5.0369e+00f,  6.0365e+00f,  6.0366e+00f,
            3.7200e-02f,  2.0371e+00f,  4.0373e+00f,  6.0370e+00f,  6.0374e+00f,  3.7800e-02f,  3.0377e+00f,  3.0379e+00f,  6.0376e+00f,  7.0375e+00f,
            1.0384e+00f,  3.0382e+00f,  4.0380e+00f,  5.0381e+00f,  7.0383e+00f,  1.0389e+00f,  3.0387e+00f,  6.0386e+00f,  7.0385e+00f,  7.0388e+00f,
            3.9200e-02f,  1.0390e+00f,  2.0391e+00f,  4.0393e+00f,  6.0394e+00f,  3.9600e-02f,  3.9900e-02f,  1.0395e+00f,  5.0397e+00f,  6.0398e+00f,
            4.0000e-02f,  4.0402e+00f,  5.0404e+00f,  6.0401e+00f,  7.0403e+00f,  1.0407e+00f,  1.0409e+00f,  4.0406e+00f,  5.0405e+00f,  7.0408e+00f,
            1.0410e+00f,  2.0412e+00f,  5.0413e+00f,  6.0414e+00f,  7.0411e+00f,  2.0418e+00f,  4.0419e+00f,  5.0416e+00f,  7.0415e+00f,  7.0417e+00f,
            2.0421e+00f,  2.0422e+00f,  4.0420e+00f,  5.0423e+00f,  6.0424e+00f,  4.2800e-02f,  2.0426e+00f,  3.0429e+00f,  4.0425e+00f,  6.0427e+00f,
            4.3400e-02f,  1.0430e+00f,  4.0432e+00f,  7.0431e+00f,  7.0433e+00f,  1.0436e+00f,  5.0437e+00f,  7.0435e+00f,  7.0438e+00f,  7.0439e+00f,
            1.0442e+00f,  3.0441e+00f,  3.0443e+00f,  4.0444e+00f,  6.0440e+00f,  4.4800e-02f,  2.0446e+00f,  3.0445e+00f,  5.0449e+00f,  6.0447e+00f,
            1.0451e+00f,  4.0450e+00f,  4.0452e+00f,  6.0453e+00f,  6.0454e+00f,  1.0457e+00f,  3.0458e+00f,  4.0455e+00f,  5.0456e+00f,  5.0459e+00f,
            1.0461e+00f,  3.0463e+00f,  3.0464e+00f,  4.0460e+00f,  4.0462e+00f,  4.6900e-02f,  1.0465e+00f,  1.0467e+00f,  2.0468e+00f,  6.0466e+00f,
            2.0473e+00f,  2.0474e+00f,  3.0471e+00f,  6.0470e+00f,  6.0472e+00f,  4.7500e-02f,  2.0476e+00f,  5.0477e+00f,  5.0479e+00f,  6.0478e+00f,
            4.8100e-02f,  2.0482e+00f,  3.0483e+00f,  3.0484e+00f,  4.0480e+00f,  1.0485e+00f,  3.0487e+00f,  3.0489e+00f,  4.0488e+00f,  5.0486e+00f,
            3.0491e+00f,  5.0492e+00f,  6.0494e+00f,  7.0490e+00f,  7.0493e+00f,  4.9600e-02f,  3.0497e+00f,  3.0499e+00f,  4.0495e+00f,  4.0498e+00f,
            5.0100e-02f,  5.0400e-02f,  2.0502e+00f,  2.0503e+00f,  4.0500e+00f,  5.0600e-02f,  4.0507e+00f,  5.0505e+00f,  5.0509e+00f,  6.0508e+00f,
            1.0512e+00f,  1.0513e+00f,  3.0510e+00f,  4.0514e+00f,  7.0511e+00f,  1.0515e+00f,  3.0519e+00f,  4.0516e+00f,  5.0517e+00f,  5.0518e+00f,
            5.2100e-02f,  2.0522e+00f,  2.0523e+00f,  2.0524e+00f,  7.0520e+00f,  5.2700e-02f,  5.2800e-02f,  2.0526e+00f,  6.0529e+00f,  7.0525e+00f,
            2.0530e+00f,  2.0534e+00f,  5.0531e+00f,  5.0533e+00f,  6.0532e+00f,  2.0536e+00f,  3.0537e+00f,  3.0539e+00f,  5.0538e+00f,  7.0535e+00f,
            5.4200e-02f,  5.0544e+00f,  7.0540e+00f,  7.0541e+00f,  7.0543e+00f,  1.0545e+00f,  2.0549e+00f,  3.0546e+00f,  4.0548e+00f,  5.0547e+00f,
            5.5400e-02f,  3.0551e+00f,  3.0553e+00f,  4.0552e+00f,  6.0550e+00f,  1.0559e+00f,  2.0555e+00f,  4.0556e+00f,  4.0557e+00f,  4.0558e+00f,
        };
        readonly float[] gx_expect = {
            -1.00000e+00f,  -9.99900e-01f,  -3.99980e+00f,  -2.99970e+00f,  4.00000e-04f,  -9.99950e+00f,  -9.99940e+00f,  -8.99930e+00f,  -7.99920e+00f,  -1.09991e+01f,
            -1.79990e+01f,  -2.19989e+01f,  -2.39988e+01f,  -1.99987e+01f,  -2.09986e+01f,  -2.89985e+01f,  -2.89984e+01f,  -3.09983e+01f,  -2.99982e+01f,  -3.09981e+01f,
            -4.19980e+01f,  -3.89979e+01f,  -4.09978e+01f,  -4.09977e+01f,  -4.19976e+01f,  -5.09975e+01f,  -4.99974e+01f,  -4.99973e+01f,  -4.99972e+01f,  -5.09971e+01f,
            -6.19970e+01f,  -5.99969e+01f,  -6.19968e+01f,  -6.19967e+01f,  -6.39966e+01f,  -6.89965e+01f,  -7.19964e+01f,  -7.09963e+01f,  -6.79962e+01f,  -7.19961e+01f,
            -7.99960e+01f,  -8.09959e+01f,  -8.29958e+01f,  -8.09957e+01f,  -8.29956e+01f,  -8.89955e+01f,  -9.39954e+01f,  -9.09953e+01f,  -9.29952e+01f,  -9.29951e+01f,
            -9.89950e+01f,  -1.00995e+02f,  -9.99948e+01f,  -9.79947e+01f,  -1.01995e+02f,  -1.08995e+02f,  -1.11994e+02f,  -1.10994e+02f,  -1.13994e+02f,  -1.11994e+02f,
            -1.19994e+02f,  -1.21994e+02f,  -1.19994e+02f,  -1.20994e+02f,  -1.21994e+02f,  -1.29994e+02f,  -1.31993e+02f,  -1.30993e+02f,  -1.28993e+02f,  -1.32993e+02f,
            -1.36993e+02f,  -1.38993e+02f,  -1.37993e+02f,  -1.40993e+02f,  -1.39993e+02f,  -1.48993e+02f,  -1.50992e+02f,  -1.48992e+02f,  -1.50992e+02f,  -1.51992e+02f,
            -1.59992e+02f,  -1.61992e+02f,  -1.63992e+02f,  -1.62992e+02f,  -1.65992e+02f,  -1.69992e+02f,  -1.71991e+02f,  -1.69991e+02f,  -1.69991e+02f,  -1.68991e+02f,
            -1.81991e+02f,  -1.80991e+02f,  -1.79991e+02f,  -1.82991e+02f,  -1.81991e+02f,  -1.88990e+02f,  -1.90990e+02f,  -1.90990e+02f,  -1.90990e+02f,  -1.92990e+02f,
            -1.99990e+02f,  -1.98990e+02f,  -1.98990e+02f,  -2.00990e+02f,  -1.98990e+02f,  -2.12989e+02f,  -2.11989e+02f,  -2.09989e+02f,  -2.13989e+02f,  -2.10989e+02f,
            -2.21989e+02f,  -2.20989e+02f,  -2.21989e+02f,  -2.19989e+02f,  -2.23989e+02f,  -2.27988e+02f,  -2.29988e+02f,  -2.28988e+02f,  -2.28988e+02f,  -2.30988e+02f,
            -2.39988e+02f,  -2.40988e+02f,  -2.39988e+02f,  -2.38988e+02f,  -2.40988e+02f,  -2.50988e+02f,  -2.47987e+02f,  -2.50987e+02f,  -2.49987e+02f,  -2.50987e+02f,
            -2.60987e+02f,  -2.60987e+02f,  -2.59987e+02f,  -2.59987e+02f,  -2.59987e+02f,  -2.69986e+02f,  -2.69986e+02f,  -2.68986e+02f,  -2.71986e+02f,  -2.72986e+02f,
            -2.81986e+02f,  -2.79986e+02f,  -2.81986e+02f,  -2.80986e+02f,  -2.78986e+02f,  -2.90986e+02f,  -2.90985e+02f,  -2.90985e+02f,  -2.89985e+02f,  -2.92985e+02f,
            -3.00985e+02f,  -3.00985e+02f,  -3.00985e+02f,  -2.98985e+02f,  -2.99985e+02f,  -3.08985e+02f,  -3.07984e+02f,  -3.08984e+02f,  -3.09984e+02f,  -3.10984e+02f,
            -3.19984e+02f,  -3.21984e+02f,  -3.19984e+02f,  -3.21984e+02f,  -3.21984e+02f,  -3.28983e+02f,  -3.31983e+02f,  -3.29983e+02f,  -3.29983e+02f,  -3.31983e+02f,
            -3.40983e+02f,  -3.38983e+02f,  -3.40983e+02f,  -3.38983e+02f,  -3.39983e+02f,  -3.50983e+02f,  -3.49982e+02f,  -3.50982e+02f,  -3.52982e+02f,  -3.54982e+02f,
            -3.62982e+02f,  -3.60982e+02f,  -3.63982e+02f,  -3.59982e+02f,  -3.61982e+02f,  -3.70981e+02f,  -3.67981e+02f,  -3.67981e+02f,  -3.69981e+02f,  -3.71981e+02f,
            -3.81981e+02f,  -3.82981e+02f,  -3.79981e+02f,  -3.80981e+02f,  -3.83981e+02f,  -3.89981e+02f,  -3.89980e+02f,  -3.88980e+02f,  -3.88980e+02f,  -3.90980e+02f,
            -3.98980e+02f,  -4.00980e+02f,  -3.99980e+02f,  -4.00980e+02f,  -4.00980e+02f,  -4.10979e+02f,  -4.08979e+02f,  -4.08979e+02f,  -4.11979e+02f,  -4.09979e+02f,
            -4.17979e+02f,  -4.20979e+02f,  -4.19979e+02f,  -4.22979e+02f,  -4.21979e+02f,  -4.33978e+02f,  -4.29978e+02f,  -4.31978e+02f,  -4.33978e+02f,  -4.30978e+02f,
            -4.35978e+02f,  -4.35978e+02f,  -4.37978e+02f,  -4.40978e+02f,  -4.39978e+02f,  -4.50978e+02f,  -4.50977e+02f,  -4.50977e+02f,  -4.49977e+02f,  -4.50977e+02f,
            -4.59977e+02f,  -4.58977e+02f,  -4.58977e+02f,  -4.60977e+02f,  -4.60977e+02f,  -4.69976e+02f,  -4.69976e+02f,  -4.69976e+02f,  -4.71976e+02f,  -4.70976e+02f,
            -4.75976e+02f,  -4.80976e+02f,  -4.77976e+02f,  -4.79976e+02f,  -4.77976e+02f,  -4.88976e+02f,  -4.91975e+02f,  -4.92975e+02f,  -4.90975e+02f,  -4.90975e+02f,
            -5.01975e+02f,  -4.98975e+02f,  -4.98975e+02f,  -5.00975e+02f,  -4.98975e+02f,  -5.10974e+02f,  -5.09974e+02f,  -5.12974e+02f,  -5.11974e+02f,  -5.10974e+02f,
            -5.16974e+02f,  -5.18974e+02f,  -5.19974e+02f,  -5.21974e+02f,  -5.17974e+02f,  -5.29973e+02f,  -5.27973e+02f,  -5.30973e+02f,  -5.29973e+02f,  -5.29973e+02f,
            -5.39973e+02f,  -5.39973e+02f,  -5.41973e+02f,  -5.40973e+02f,  -5.40973e+02f,  -5.46972e+02f,  -5.45972e+02f,  -5.48972e+02f,  -5.47972e+02f,  -5.50972e+02f,
            -5.58972e+02f,  -5.60972e+02f,  -5.62972e+02f,  -5.60972e+02f,  -5.59972e+02f,  -5.70971e+02f,  -5.70971e+02f,  -5.70971e+02f,  -5.69971e+02f,  -5.69971e+02f,
            -5.77971e+02f,  -5.77971e+02f,  -5.79971e+02f,  -5.80971e+02f,  -5.81971e+02f,  -5.90971e+02f,  -5.89970e+02f,  -5.91970e+02f,  -5.89970e+02f,  -5.91970e+02f,
            -5.98970e+02f,  -5.98970e+02f,  -6.00970e+02f,  -6.02970e+02f,  -5.97970e+02f,  -6.10970e+02f,  -6.08969e+02f,  -6.10969e+02f,  -6.08969e+02f,  -6.12969e+02f,
            -6.19969e+02f,  -6.22969e+02f,  -6.21969e+02f,  -6.22969e+02f,  -6.23969e+02f,  -6.28968e+02f,  -6.29968e+02f,  -6.29968e+02f,  -6.31968e+02f,  -6.29968e+02f,
            -6.39968e+02f,  -6.42968e+02f,  -6.41968e+02f,  -6.39968e+02f,  -6.43968e+02f,  -6.49967e+02f,  -6.49967e+02f,  -6.50967e+02f,  -6.48967e+02f,  -6.50967e+02f,
            -6.57967e+02f,  -6.60967e+02f,  -6.59967e+02f,  -6.56967e+02f,  -6.56967e+02f,  -6.66966e+02f,  -6.68966e+02f,  -6.70966e+02f,  -6.67966e+02f,  -6.67966e+02f,
            -6.79966e+02f,  -6.82966e+02f,  -6.81966e+02f,  -6.80966e+02f,  -6.79966e+02f,  -6.87966e+02f,  -6.89965e+02f,  -6.88965e+02f,  -6.89965e+02f,  -6.90965e+02f,
            -6.99965e+02f,  -7.01965e+02f,  -7.01965e+02f,  -7.00965e+02f,  -7.02965e+02f,  -7.08965e+02f,  -7.10964e+02f,  -7.08964e+02f,  -7.07964e+02f,  -7.10964e+02f,
            -7.21964e+02f,  -7.20964e+02f,  -7.20964e+02f,  -7.22964e+02f,  -7.19964e+02f,  -7.29963e+02f,  -7.31963e+02f,  -7.27963e+02f,  -7.27963e+02f,  -7.28963e+02f,
            -7.39963e+02f,  -7.39963e+02f,  -7.39963e+02f,  -7.39963e+02f,  -7.41963e+02f,  -7.50962e+02f,  -7.49962e+02f,  -7.48962e+02f,  -7.49962e+02f,  -7.50962e+02f,
            -7.59962e+02f,  -7.60962e+02f,  -7.58962e+02f,  -7.60962e+02f,  -7.58962e+02f,  -7.68962e+02f,  -7.67961e+02f,  -7.68961e+02f,  -7.70961e+02f,  -7.68961e+02f,
            -7.80961e+02f,  -7.81961e+02f,  -7.79961e+02f,  -7.81961e+02f,  -7.81961e+02f,  -7.92961e+02f,  -7.89960e+02f,  -7.90960e+02f,  -7.91960e+02f,  -7.91960e+02f,
            -7.99960e+02f,  -7.99960e+02f,  -7.97960e+02f,  -8.00960e+02f,  -7.98960e+02f,  -8.10960e+02f,  -8.09959e+02f,  -8.08959e+02f,  -8.10959e+02f,  -8.10959e+02f,
            -8.18959e+02f,  -8.20959e+02f,  -8.19959e+02f,  -8.18959e+02f,  -8.19959e+02f,  -8.28958e+02f,  -8.28958e+02f,  -8.30958e+02f,  -8.27958e+02f,  -8.27958e+02f,
            -8.39958e+02f,  -8.37958e+02f,  -8.39958e+02f,  -8.40958e+02f,  -8.41958e+02f,  -8.51957e+02f,  -8.49957e+02f,  -8.51957e+02f,  -8.49957e+02f,  -8.50957e+02f,
            -8.60957e+02f,  -8.58957e+02f,  -8.59957e+02f,  -8.60957e+02f,  -8.59957e+02f,  -8.66957e+02f,  -8.68956e+02f,  -8.66956e+02f,  -8.68956e+02f,  -8.70956e+02f,
            -8.81956e+02f,  -8.78956e+02f,  -8.78956e+02f,  -8.80956e+02f,  -8.81956e+02f,  -8.90956e+02f,  -8.89955e+02f,  -8.91955e+02f,  -8.89955e+02f,  -8.90955e+02f,
            -8.97955e+02f,  -8.98955e+02f,  -8.99955e+02f,  -8.99955e+02f,  -9.01955e+02f,  -9.09955e+02f,  -9.10954e+02f,  -9.08954e+02f,  -9.08954e+02f,  -9.12954e+02f,
            -9.21954e+02f,  -9.18954e+02f,  -9.23954e+02f,  -9.18954e+02f,  -9.20954e+02f,  -9.30953e+02f,  -9.31953e+02f,  -9.32953e+02f,  -9.33953e+02f,  -9.29953e+02f,
            -9.39953e+02f,  -9.40953e+02f,  -9.41953e+02f,  -9.37953e+02f,  -9.39953e+02f,  -9.49952e+02f,  -9.49952e+02f,  -9.48952e+02f,  -9.51952e+02f,  -9.50952e+02f,
            -9.63952e+02f,  -9.59952e+02f,  -9.59952e+02f,  -9.60952e+02f,  -9.62952e+02f,  -9.68952e+02f,  -9.72951e+02f,  -9.68951e+02f,  -9.71951e+02f,  -9.70951e+02f,
            -9.78951e+02f,  -9.76951e+02f,  -9.76951e+02f,  -9.80951e+02f,  -9.77951e+02f,  -9.91951e+02f,  -9.89950e+02f,  -9.88950e+02f,  -9.93950e+02f,  -9.90950e+02f,
            -1.00395e+03f,  -9.99950e+02f,  -1.00195e+03f,  -1.00395e+03f,  -1.00195e+03f,  -1.00895e+03f,  -1.00995e+03f,  -1.00795e+03f,  -1.01195e+03f,  -1.01095e+03f,
            -1.02095e+03f,  -1.02095e+03f,  -1.01895e+03f,  -1.02095e+03f,  -1.02195e+03f,  -1.02895e+03f,  -1.02995e+03f,  -1.03095e+03f,  -1.03295e+03f,  -1.02895e+03f,
            -1.04095e+03f,  -1.03995e+03f,  -1.03995e+03f,  -1.04195e+03f,  -1.04395e+03f,  -1.05095e+03f,  -1.05195e+03f,  -1.04995e+03f,  -1.05195e+03f,  -1.04995e+03f,
            -1.05795e+03f,  -1.05895e+03f,  -1.06195e+03f,  -1.06095e+03f,  -1.05995e+03f,  -1.07095e+03f,  -1.06795e+03f,  -1.06895e+03f,  -1.07095e+03f,  -1.07095e+03f,
            -1.07695e+03f,  -1.07895e+03f,  -1.07995e+03f,  -1.08095e+03f,  -1.07695e+03f,  -1.08895e+03f,  -1.09095e+03f,  -1.09295e+03f,  -1.09195e+03f,  -1.08995e+03f,
            -1.10194e+03f,  -1.09894e+03f,  -1.10194e+03f,  -1.10094e+03f,  -1.09994e+03f,  -1.10994e+03f,  -1.10994e+03f,  -1.11194e+03f,  -1.11394e+03f,  -1.10894e+03f,
        };
    }
}