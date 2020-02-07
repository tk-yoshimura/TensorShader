using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ArrayManipulation {
    [TestClass]
    public class ConcatTest {
        [TestMethod]
        public void ReferenceTest() {
            int ch1 = 2, ch2 = 1, ch3 = 4, ch4 = 3, chsum = ch1 + ch2 + ch3 + ch4;
            int s1 = 6, s2 = 4, s3 = 5;

            float[] x1val = (new float[ch1 * s1 * s2 * s3]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] x2val = (new float[ch2 * s1 * s2 * s3]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] x3val = (new float[ch3 * s1 * s2 * s3]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] x4val = (new float[ch4 * s1 * s2 * s3]).Select((_, idx) => idx * 1e-3f).ToArray();

            float[] yval = (new float[chsum * s1 * s2 * s3]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor x1tensor = new Tensor(new Shape(ShapeType.Map, ch1, s1, s2, s3), x1val);
            Tensor x2tensor = new Tensor(new Shape(ShapeType.Map, ch2, s1, s2, s3), x2val);
            Tensor x3tensor = new Tensor(new Shape(ShapeType.Map, ch3, s1, s2, s3), x3val);
            Tensor x4tensor = new Tensor(new Shape(ShapeType.Map, ch4, s1, s2, s3), x4val);

            Tensor ytensor = new Tensor(new Shape(ShapeType.Map, chsum, s1, s2, s3), yval);

            ParameterField x1 = x1tensor;
            ParameterField x2 = x2tensor;
            VariableField x3 = x3tensor;
            ParameterField x4 = x4tensor;
            VariableField y_actual = ytensor;

            Field y_expect = Concat(axis: 0, x1, x2, x3, x4);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx1_actual = x1.GradState;

            AssertError.Tolerance(gx1_expect, gx1_actual, 1e-7f, 1e-5f, $"not equal gx1");

            float[] gx2_actual = x2.GradState;

            AssertError.Tolerance(gx2_expect, gx2_actual, 1e-7f, 1e-5f, $"not equal gx2");

            float[] gx4_actual = x4.GradState;

            AssertError.Tolerance(gx4_expect, gx4_actual, 1e-7f, 1e-5f, $"not equal gx4");
        }

        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            foreach (Shape outshape in new Shape[]{
                new Shape(ShapeType.Map, 13),
                new Shape(ShapeType.Map, 24),
                new Shape(ShapeType.Map, 13, 14),
                new Shape(ShapeType.Map, 24, 21),
                new Shape(ShapeType.Map, 14, 19, 13),
                new Shape(ShapeType.Map, 13, 14, 19),
                new Shape(ShapeType.Map, 13, 14, 19, 21),
                new Shape(ShapeType.Map, 19, 21, 13, 14),
                new Shape(ShapeType.Map, 13, 14, 19, 21, 24),
                new Shape(ShapeType.Map, 24, 19, 14, 13, 21),
                new Shape(ShapeType.Map, 19, 13, 21, 24, 14)}) {
                for (int axis = 0; axis < outshape.Ndim; axis++) {
                    int length = outshape[axis];

                    for (int n = 1; n <= 5; n++) {
                        int[] c = (new int[n]).Select((_) => 1).ToArray();

                        for (int j = n; j < length; j++) {
                            c[rd.Next(c.Length)]++;
                        }

                        float[][] xs = new float[n][];
                        Shape[] inshapes = new Shape[n];
                        ParameterField[] xfields = new ParameterField[n];

                        for (int i = 0; i < n; i++) {
                            int[] s = outshape;
                            s[axis] = c[i];
                            inshapes[i] = new Shape(ShapeType.Map, s);

                            xs[i] = (new float[inshapes[i].Length]).Select((_) => (float)rd.NextDouble() * (i + 1)).ToArray();

                            xfields[i] = new OverflowCheckedTensor(inshapes[i], xs[i]);
                        }

                        VariableField tfield = new OverflowCheckedTensor(outshape);

                        Field yfield = Concat(axis, xfields);

                        Field err = yfield - tfield;

                        (Flow flow, Parameters parameters) = Flow.Optimize(err);

                        flow.Execute();

                        Console.WriteLine($"pass {outshape} axis:{axis} <- {string.Join(", ", inshapes.Select((shape) => shape.ToString()))}");
                    }
                };
            }
        }

        float[] gx1_expect = {
            -1.1990e+00f, -1.1970e+00f, -1.1870e+00f, -1.1850e+00f, -1.1750e+00f,
            -1.1730e+00f, -1.1630e+00f, -1.1610e+00f, -1.1510e+00f, -1.1490e+00f,
            -1.1390e+00f, -1.1370e+00f, -1.1270e+00f, -1.1250e+00f, -1.1150e+00f,
            -1.1130e+00f, -1.1030e+00f, -1.1010e+00f, -1.0910e+00f, -1.0890e+00f,
            -1.0790e+00f, -1.0770e+00f, -1.0670e+00f, -1.0650e+00f, -1.0550e+00f,
            -1.0530e+00f, -1.0430e+00f, -1.0410e+00f, -1.0310e+00f, -1.0290e+00f,
            -1.0190e+00f, -1.0170e+00f, -1.0070e+00f, -1.0050e+00f, -9.9500e-01f,
            -9.9300e-01f, -9.8300e-01f, -9.8100e-01f, -9.7100e-01f, -9.6900e-01f,
            -9.5900e-01f, -9.5700e-01f, -9.4700e-01f, -9.4500e-01f, -9.3500e-01f,
            -9.3300e-01f, -9.2300e-01f, -9.2100e-01f, -9.1100e-01f, -9.0900e-01f,
            -8.9900e-01f, -8.9700e-01f, -8.8700e-01f, -8.8500e-01f, -8.7500e-01f,
            -8.7300e-01f, -8.6300e-01f, -8.6100e-01f, -8.5100e-01f, -8.4900e-01f,
            -8.3900e-01f, -8.3700e-01f, -8.2700e-01f, -8.2500e-01f, -8.1500e-01f,
            -8.1300e-01f, -8.0300e-01f, -8.0100e-01f, -7.9100e-01f, -7.8900e-01f,
            -7.7900e-01f, -7.7700e-01f, -7.6700e-01f, -7.6500e-01f, -7.5500e-01f,
            -7.5300e-01f, -7.4300e-01f, -7.4100e-01f, -7.3100e-01f, -7.2900e-01f,
            -7.1900e-01f, -7.1700e-01f, -7.0700e-01f, -7.0500e-01f, -6.9500e-01f,
            -6.9300e-01f, -6.8300e-01f, -6.8100e-01f, -6.7100e-01f, -6.6900e-01f,
            -6.5900e-01f, -6.5700e-01f, -6.4700e-01f, -6.4500e-01f, -6.3500e-01f,
            -6.3300e-01f, -6.2300e-01f, -6.2100e-01f, -6.1100e-01f, -6.0900e-01f,
            -5.9900e-01f, -5.9700e-01f, -5.8700e-01f, -5.8500e-01f, -5.7500e-01f,
            -5.7300e-01f, -5.6300e-01f, -5.6100e-01f, -5.5100e-01f, -5.4900e-01f,
            -5.3900e-01f, -5.3700e-01f, -5.2700e-01f, -5.2500e-01f, -5.1500e-01f,
            -5.1300e-01f, -5.0300e-01f, -5.0100e-01f, -4.9100e-01f, -4.8900e-01f,
            -4.7900e-01f, -4.7700e-01f, -4.6700e-01f, -4.6500e-01f, -4.5500e-01f,
            -4.5300e-01f, -4.4300e-01f, -4.4100e-01f, -4.3100e-01f, -4.2900e-01f,
            -4.1900e-01f, -4.1700e-01f, -4.0700e-01f, -4.0500e-01f, -3.9500e-01f,
            -3.9300e-01f, -3.8300e-01f, -3.8100e-01f, -3.7100e-01f, -3.6900e-01f,
            -3.5900e-01f, -3.5700e-01f, -3.4700e-01f, -3.4500e-01f, -3.3500e-01f,
            -3.3300e-01f, -3.2300e-01f, -3.2100e-01f, -3.1100e-01f, -3.0900e-01f,
            -2.9900e-01f, -2.9700e-01f, -2.8700e-01f, -2.8500e-01f, -2.7500e-01f,
            -2.7300e-01f, -2.6300e-01f, -2.6100e-01f, -2.5100e-01f, -2.4900e-01f,
            -2.3900e-01f, -2.3700e-01f, -2.2700e-01f, -2.2500e-01f, -2.1500e-01f,
            -2.1300e-01f, -2.0300e-01f, -2.0100e-01f, -1.9100e-01f, -1.8900e-01f,
            -1.7900e-01f, -1.7700e-01f, -1.6700e-01f, -1.6500e-01f, -1.5500e-01f,
            -1.5300e-01f, -1.4300e-01f, -1.4100e-01f, -1.3100e-01f, -1.2900e-01f,
            -1.1900e-01f, -1.1700e-01f, -1.0700e-01f, -1.0500e-01f, -9.5000e-02f,
            -9.3000e-02f, -8.3000e-02f, -8.1000e-02f, -7.1000e-02f, -6.9000e-02f,
            -5.9000e-02f, -5.7000e-02f, -4.7000e-02f, -4.5000e-02f, -3.5000e-02f,
            -3.3000e-02f, -2.3000e-02f, -2.1000e-02f, -1.1000e-02f, -9.0000e-03f,
            1.0000e-03f, 3.0000e-03f, 1.3000e-02f, 1.5000e-02f, 2.5000e-02f,
            2.7000e-02f, 3.7000e-02f, 3.9000e-02f, 4.9000e-02f, 5.1000e-02f,
            6.1000e-02f, 6.3000e-02f, 7.3000e-02f, 7.5000e-02f, 8.5000e-02f,
            8.7000e-02f, 9.7000e-02f, 9.9000e-02f, 1.0900e-01f, 1.1100e-01f,
            1.2100e-01f, 1.2300e-01f, 1.3300e-01f, 1.3500e-01f, 1.4500e-01f,
            1.4700e-01f, 1.5700e-01f, 1.5900e-01f, 1.6900e-01f, 1.7100e-01f,
            1.8100e-01f, 1.8300e-01f, 1.9300e-01f, 1.9500e-01f, 2.0500e-01f,
            2.0700e-01f, 2.1700e-01f, 2.1900e-01f, 2.2900e-01f, 2.3100e-01f,
        };

        float[] gx2_expect = {
            -1.1970e+00f, -1.1860e+00f, -1.1750e+00f, -1.1640e+00f, -1.1530e+00f,
            -1.1420e+00f, -1.1310e+00f, -1.1200e+00f, -1.1090e+00f, -1.0980e+00f,
            -1.0870e+00f, -1.0760e+00f, -1.0650e+00f, -1.0540e+00f, -1.0430e+00f,
            -1.0320e+00f, -1.0210e+00f, -1.0100e+00f, -9.9900e-01f, -9.8800e-01f,
            -9.7700e-01f, -9.6600e-01f, -9.5500e-01f, -9.4400e-01f, -9.3300e-01f,
            -9.2200e-01f, -9.1100e-01f, -9.0000e-01f, -8.8900e-01f, -8.7800e-01f,
            -8.6700e-01f, -8.5600e-01f, -8.4500e-01f, -8.3400e-01f, -8.2300e-01f,
            -8.1200e-01f, -8.0100e-01f, -7.9000e-01f, -7.7900e-01f, -7.6800e-01f,
            -7.5700e-01f, -7.4600e-01f, -7.3500e-01f, -7.2400e-01f, -7.1300e-01f,
            -7.0200e-01f, -6.9100e-01f, -6.8000e-01f, -6.6900e-01f, -6.5800e-01f,
            -6.4700e-01f, -6.3600e-01f, -6.2500e-01f, -6.1400e-01f, -6.0300e-01f,
            -5.9200e-01f, -5.8100e-01f, -5.7000e-01f, -5.5900e-01f, -5.4800e-01f,
            -5.3700e-01f, -5.2600e-01f, -5.1500e-01f, -5.0400e-01f, -4.9300e-01f,
            -4.8200e-01f, -4.7100e-01f, -4.6000e-01f, -4.4900e-01f, -4.3800e-01f,
            -4.2700e-01f, -4.1600e-01f, -4.0500e-01f, -3.9400e-01f, -3.8300e-01f,
            -3.7200e-01f, -3.6100e-01f, -3.5000e-01f, -3.3900e-01f, -3.2800e-01f,
            -3.1700e-01f, -3.0600e-01f, -2.9500e-01f, -2.8400e-01f, -2.7300e-01f,
            -2.6200e-01f, -2.5100e-01f, -2.4000e-01f, -2.2900e-01f, -2.1800e-01f,
            -2.0700e-01f, -1.9600e-01f, -1.8500e-01f, -1.7400e-01f, -1.6300e-01f,
            -1.5200e-01f, -1.4100e-01f, -1.3000e-01f, -1.1900e-01f, -1.0800e-01f,
            -9.7000e-02f, -8.6000e-02f, -7.5000e-02f, -6.4000e-02f, -5.3000e-02f,
            -4.2000e-02f, -3.1000e-02f, -2.0000e-02f, -9.0000e-03f, 2.0000e-03f,
            1.3000e-02f, 2.4000e-02f, 3.5000e-02f, 4.6000e-02f, 5.7000e-02f,
            6.8000e-02f, 7.9000e-02f, 9.0000e-02f, 1.0100e-01f, 1.1200e-01f,
        };

        float[] gx4_expect = {
            -1.1920e+00f, -1.1900e+00f, -1.1880e+00f, -1.1790e+00f, -1.1770e+00f,
            -1.1750e+00f, -1.1660e+00f, -1.1640e+00f, -1.1620e+00f, -1.1530e+00f,
            -1.1510e+00f, -1.1490e+00f, -1.1400e+00f, -1.1380e+00f, -1.1360e+00f,
            -1.1270e+00f, -1.1250e+00f, -1.1230e+00f, -1.1140e+00f, -1.1120e+00f,
            -1.1100e+00f, -1.1010e+00f, -1.0990e+00f, -1.0970e+00f, -1.0880e+00f,
            -1.0860e+00f, -1.0840e+00f, -1.0750e+00f, -1.0730e+00f, -1.0710e+00f,
            -1.0620e+00f, -1.0600e+00f, -1.0580e+00f, -1.0490e+00f, -1.0470e+00f,
            -1.0450e+00f, -1.0360e+00f, -1.0340e+00f, -1.0320e+00f, -1.0230e+00f,
            -1.0210e+00f, -1.0190e+00f, -1.0100e+00f, -1.0080e+00f, -1.0060e+00f,
            -9.9700e-01f, -9.9500e-01f, -9.9300e-01f, -9.8400e-01f, -9.8200e-01f,
            -9.8000e-01f, -9.7100e-01f, -9.6900e-01f, -9.6700e-01f, -9.5800e-01f,
            -9.5600e-01f, -9.5400e-01f, -9.4500e-01f, -9.4300e-01f, -9.4100e-01f,
            -9.3200e-01f, -9.3000e-01f, -9.2800e-01f, -9.1900e-01f, -9.1700e-01f,
            -9.1500e-01f, -9.0600e-01f, -9.0400e-01f, -9.0200e-01f, -8.9300e-01f,
            -8.9100e-01f, -8.8900e-01f, -8.8000e-01f, -8.7800e-01f, -8.7600e-01f,
            -8.6700e-01f, -8.6500e-01f, -8.6300e-01f, -8.5400e-01f, -8.5200e-01f,
            -8.5000e-01f, -8.4100e-01f, -8.3900e-01f, -8.3700e-01f, -8.2800e-01f,
            -8.2600e-01f, -8.2400e-01f, -8.1500e-01f, -8.1300e-01f, -8.1100e-01f,
            -8.0200e-01f, -8.0000e-01f, -7.9800e-01f, -7.8900e-01f, -7.8700e-01f,
            -7.8500e-01f, -7.7600e-01f, -7.7400e-01f, -7.7200e-01f, -7.6300e-01f,
            -7.6100e-01f, -7.5900e-01f, -7.5000e-01f, -7.4800e-01f, -7.4600e-01f,
            -7.3700e-01f, -7.3500e-01f, -7.3300e-01f, -7.2400e-01f, -7.2200e-01f,
            -7.2000e-01f, -7.1100e-01f, -7.0900e-01f, -7.0700e-01f, -6.9800e-01f,
            -6.9600e-01f, -6.9400e-01f, -6.8500e-01f, -6.8300e-01f, -6.8100e-01f,
            -6.7200e-01f, -6.7000e-01f, -6.6800e-01f, -6.5900e-01f, -6.5700e-01f,
            -6.5500e-01f, -6.4600e-01f, -6.4400e-01f, -6.4200e-01f, -6.3300e-01f,
            -6.3100e-01f, -6.2900e-01f, -6.2000e-01f, -6.1800e-01f, -6.1600e-01f,
            -6.0700e-01f, -6.0500e-01f, -6.0300e-01f, -5.9400e-01f, -5.9200e-01f,
            -5.9000e-01f, -5.8100e-01f, -5.7900e-01f, -5.7700e-01f, -5.6800e-01f,
            -5.6600e-01f, -5.6400e-01f, -5.5500e-01f, -5.5300e-01f, -5.5100e-01f,
            -5.4200e-01f, -5.4000e-01f, -5.3800e-01f, -5.2900e-01f, -5.2700e-01f,
            -5.2500e-01f, -5.1600e-01f, -5.1400e-01f, -5.1200e-01f, -5.0300e-01f,
            -5.0100e-01f, -4.9900e-01f, -4.9000e-01f, -4.8800e-01f, -4.8600e-01f,
            -4.7700e-01f, -4.7500e-01f, -4.7300e-01f, -4.6400e-01f, -4.6200e-01f,
            -4.6000e-01f, -4.5100e-01f, -4.4900e-01f, -4.4700e-01f, -4.3800e-01f,
            -4.3600e-01f, -4.3400e-01f, -4.2500e-01f, -4.2300e-01f, -4.2100e-01f,
            -4.1200e-01f, -4.1000e-01f, -4.0800e-01f, -3.9900e-01f, -3.9700e-01f,
            -3.9500e-01f, -3.8600e-01f, -3.8400e-01f, -3.8200e-01f, -3.7300e-01f,
            -3.7100e-01f, -3.6900e-01f, -3.6000e-01f, -3.5800e-01f, -3.5600e-01f,
            -3.4700e-01f, -3.4500e-01f, -3.4300e-01f, -3.3400e-01f, -3.3200e-01f,
            -3.3000e-01f, -3.2100e-01f, -3.1900e-01f, -3.1700e-01f, -3.0800e-01f,
            -3.0600e-01f, -3.0400e-01f, -2.9500e-01f, -2.9300e-01f, -2.9100e-01f,
            -2.8200e-01f, -2.8000e-01f, -2.7800e-01f, -2.6900e-01f, -2.6700e-01f,
            -2.6500e-01f, -2.5600e-01f, -2.5400e-01f, -2.5200e-01f, -2.4300e-01f,
            -2.4100e-01f, -2.3900e-01f, -2.3000e-01f, -2.2800e-01f, -2.2600e-01f,
            -2.1700e-01f, -2.1500e-01f, -2.1300e-01f, -2.0400e-01f, -2.0200e-01f,
            -2.0000e-01f, -1.9100e-01f, -1.8900e-01f, -1.8700e-01f, -1.7800e-01f,
            -1.7600e-01f, -1.7400e-01f, -1.6500e-01f, -1.6300e-01f, -1.6100e-01f,
            -1.5200e-01f, -1.5000e-01f, -1.4800e-01f, -1.3900e-01f, -1.3700e-01f,
            -1.3500e-01f, -1.2600e-01f, -1.2400e-01f, -1.2200e-01f, -1.1300e-01f,
            -1.1100e-01f, -1.0900e-01f, -1.0000e-01f, -9.8000e-02f, -9.6000e-02f,
            -8.7000e-02f, -8.5000e-02f, -8.3000e-02f, -7.4000e-02f, -7.2000e-02f,
            -7.0000e-02f, -6.1000e-02f, -5.9000e-02f, -5.7000e-02f, -4.8000e-02f,
            -4.6000e-02f, -4.4000e-02f, -3.5000e-02f, -3.3000e-02f, -3.1000e-02f,
            -2.2000e-02f, -2.0000e-02f, -1.8000e-02f, -9.0000e-03f, -7.0000e-03f,
            -5.0000e-03f, 4.0000e-03f, 6.0000e-03f, 8.0000e-03f, 1.7000e-02f,
            1.9000e-02f, 2.1000e-02f, 3.0000e-02f, 3.2000e-02f, 3.4000e-02f,
            4.3000e-02f, 4.5000e-02f, 4.7000e-02f, 5.6000e-02f, 5.8000e-02f,
            6.0000e-02f, 6.9000e-02f, 7.1000e-02f, 7.3000e-02f, 8.2000e-02f,
            8.4000e-02f, 8.6000e-02f, 9.5000e-02f, 9.7000e-02f, 9.9000e-02f,
            1.0800e-01f, 1.1000e-01f, 1.1200e-01f, 1.2100e-01f, 1.2300e-01f,
            1.2500e-01f, 1.3400e-01f, 1.3600e-01f, 1.3800e-01f, 1.4700e-01f,
            1.4900e-01f, 1.5100e-01f, 1.6000e-01f, 1.6200e-01f, 1.6400e-01f,
            1.7300e-01f, 1.7500e-01f, 1.7700e-01f, 1.8600e-01f, 1.8800e-01f,
            1.9000e-01f, 1.9900e-01f, 2.0100e-01f, 2.0300e-01f, 2.1200e-01f,
            2.1400e-01f, 2.1600e-01f, 2.2500e-01f, 2.2700e-01f, 2.2900e-01f,
            2.3800e-01f, 2.4000e-01f, 2.4200e-01f, 2.5100e-01f, 2.5300e-01f,
            2.5500e-01f, 2.6400e-01f, 2.6600e-01f, 2.6800e-01f, 2.7700e-01f,
            2.7900e-01f, 2.8100e-01f, 2.9000e-01f, 2.9200e-01f, 2.9400e-01f,
            3.0300e-01f, 3.0500e-01f, 3.0700e-01f, 3.1600e-01f, 3.1800e-01f,
            3.2000e-01f, 3.2900e-01f, 3.3100e-01f, 3.3300e-01f, 3.4200e-01f,
            3.4400e-01f, 3.4600e-01f, 3.5500e-01f, 3.5700e-01f, 3.5900e-01f,
        };
    }
}
