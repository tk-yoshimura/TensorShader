using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class PointwisewiseConvolutionTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, width = 13, batch = 2;

            float[] xval = (new float[width * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[width * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, width, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, width, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel0D(inchannels, outchannels), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field y_expect = PointwiseConvolution1D(x, w);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            -1.0245400e-03f,  -9.7801000e-04f,  -9.3148000e-04f,  -8.8495000e-04f,  -8.3842000e-04f,  -7.9189000e-04f,  -7.4536000e-04f,
            -4.8816680e-03f,  -4.7346200e-03f,  -4.5875720e-03f,  -4.4405240e-03f,  -4.2934760e-03f,  -4.1464280e-03f,  -3.9993800e-03f,
            -8.7387960e-03f,  -8.4912300e-03f,  -8.2436640e-03f,  -7.9960980e-03f,  -7.7485320e-03f,  -7.5009660e-03f,  -7.2534000e-03f,
            -1.2595924e-02f,  -1.2247840e-02f,  -1.1899756e-02f,  -1.1551672e-02f,  -1.1203588e-02f,  -1.0855504e-02f,  -1.0507420e-02f,
            -1.6453052e-02f,  -1.6004450e-02f,  -1.5555848e-02f,  -1.5107246e-02f,  -1.4658644e-02f,  -1.4210042e-02f,  -1.3761440e-02f,
            -2.0310180e-02f,  -1.9761060e-02f,  -1.9211940e-02f,  -1.8662820e-02f,  -1.8113700e-02f,  -1.7564580e-02f,  -1.7015460e-02f,
            -2.4167308e-02f,  -2.3517670e-02f,  -2.2868032e-02f,  -2.2218394e-02f,  -2.1568756e-02f,  -2.0919118e-02f,  -2.0269480e-02f,
            -2.8024436e-02f,  -2.7274280e-02f,  -2.6524124e-02f,  -2.5773968e-02f,  -2.5023812e-02f,  -2.4273656e-02f,  -2.3523500e-02f,
            -3.1881564e-02f,  -3.1030890e-02f,  -3.0180216e-02f,  -2.9329542e-02f,  -2.8478868e-02f,  -2.7628194e-02f,  -2.6777520e-02f,
            -3.5738692e-02f,  -3.4787500e-02f,  -3.3836308e-02f,  -3.2885116e-02f,  -3.1933924e-02f,  -3.0982732e-02f,  -3.0031540e-02f,
            -3.9595820e-02f,  -3.8544110e-02f,  -3.7492400e-02f,  -3.6440690e-02f,  -3.5388980e-02f,  -3.4337270e-02f,  -3.3285560e-02f,
            -4.3452948e-02f,  -4.2300720e-02f,  -4.1148492e-02f,  -3.9996264e-02f,  -3.8844036e-02f,  -3.7691808e-02f,  -3.6539580e-02f,
            -4.7310076e-02f,  -4.6057330e-02f,  -4.4804584e-02f,  -4.3551838e-02f,  -4.2299092e-02f,  -4.1046346e-02f,  -3.9793600e-02f,
            -5.1167204e-02f,  -4.9813940e-02f,  -4.8460676e-02f,  -4.7107412e-02f,  -4.5754148e-02f,  -4.4400884e-02f,  -4.3047620e-02f,
            -5.5024332e-02f,  -5.3570550e-02f,  -5.2116768e-02f,  -5.0662986e-02f,  -4.9209204e-02f,  -4.7755422e-02f,  -4.6301640e-02f,
            -5.8881460e-02f,  -5.7327160e-02f,  -5.5772860e-02f,  -5.4218560e-02f,  -5.2664260e-02f,  -5.1109960e-02f,  -4.9555660e-02f,
            -6.2738588e-02f,  -6.1083770e-02f,  -5.9428952e-02f,  -5.7774134e-02f,  -5.6119316e-02f,  -5.4464498e-02f,  -5.2809680e-02f,
            -6.6595716e-02f,  -6.4840380e-02f,  -6.3085044e-02f,  -6.1329708e-02f,  -5.9574372e-02f,  -5.7819036e-02f,  -5.6063700e-02f,
            -7.0452844e-02f,  -6.8596990e-02f,  -6.6741136e-02f,  -6.4885282e-02f,  -6.3029428e-02f,  -6.1173574e-02f,  -5.9317720e-02f,
            -7.4309972e-02f,  -7.2353600e-02f,  -7.0397228e-02f,  -6.8440856e-02f,  -6.6484484e-02f,  -6.4528112e-02f,  -6.2571740e-02f,
            -7.8167100e-02f,  -7.6110210e-02f,  -7.4053320e-02f,  -7.1996430e-02f,  -6.9939540e-02f,  -6.7882650e-02f,  -6.5825760e-02f,
            -8.2024228e-02f,  -7.9866820e-02f,  -7.7709412e-02f,  -7.5552004e-02f,  -7.3394596e-02f,  -7.1237188e-02f,  -6.9079780e-02f,
            -8.5881356e-02f,  -8.3623430e-02f,  -8.1365504e-02f,  -7.9107578e-02f,  -7.6849652e-02f,  -7.4591726e-02f,  -7.2333800e-02f,
            -8.9738484e-02f,  -8.7380040e-02f,  -8.5021596e-02f,  -8.2663152e-02f,  -8.0304708e-02f,  -7.7946264e-02f,  -7.5587820e-02f,
            -9.3595612e-02f,  -9.1136650e-02f,  -8.8677688e-02f,  -8.6218726e-02f,  -8.3759764e-02f,  -8.1300802e-02f,  -7.8841840e-02f,
            -9.7452740e-02f,  -9.4893260e-02f,  -9.2333780e-02f,  -8.9774300e-02f,  -8.7214820e-02f,  -8.4655340e-02f,  -8.2095860e-02f,
        };

        float[] gw_expect = new float[] {
            -2.8366065e-01f,  -2.8603399e-01f,  -2.8840734e-01f,  -2.9078068e-01f,  -2.9315403e-01f,  -2.9552737e-01f,  -2.9790072e-01f,
            -2.9953560e-01f,  -3.0205024e-01f,  -3.0456488e-01f,  -3.0707953e-01f,  -3.0959417e-01f,  -3.1210881e-01f,  -3.1462345e-01f,
            -3.1541055e-01f,  -3.1806649e-01f,  -3.2072243e-01f,  -3.2337837e-01f,  -3.2603431e-01f,  -3.2869024e-01f,  -3.3134618e-01f,
            -3.3128550e-01f,  -3.3408274e-01f,  -3.3687997e-01f,  -3.3967721e-01f,  -3.4247444e-01f,  -3.4527168e-01f,  -3.4806892e-01f,
            -3.4716045e-01f,  -3.5009898e-01f,  -3.5303752e-01f,  -3.5597605e-01f,  -3.5891458e-01f,  -3.6185311e-01f,  -3.6479165e-01f,
            -3.6303540e-01f,  -3.6611523e-01f,  -3.6919506e-01f,  -3.7227489e-01f,  -3.7535472e-01f,  -3.7843455e-01f,  -3.8151438e-01f,
            -3.7891035e-01f,  -3.8213148e-01f,  -3.8535260e-01f,  -3.8857373e-01f,  -3.9179486e-01f,  -3.9501599e-01f,  -3.9823711e-01f,
            -3.9478530e-01f,  -3.9814772e-01f,  -4.0151015e-01f,  -4.0487257e-01f,  -4.0823500e-01f,  -4.1159742e-01f,  -4.1495984e-01f,
            -4.1066025e-01f,  -4.1416397e-01f,  -4.1766769e-01f,  -4.2117141e-01f,  -4.2467513e-01f,  -4.2817886e-01f,  -4.3168258e-01f,
            -4.2653520e-01f,  -4.3018022e-01f,  -4.3382524e-01f,  -4.3747025e-01f,  -4.4111527e-01f,  -4.4476029e-01f,  -4.4840531e-01f,
            -4.4241015e-01f,  -4.4619646e-01f,  -4.4998278e-01f,  -4.5376910e-01f,  -4.5755541e-01f,  -4.6134173e-01f,  -4.6512804e-01f,
        };
    }
}
