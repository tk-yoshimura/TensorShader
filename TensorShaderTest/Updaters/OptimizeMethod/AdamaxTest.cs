using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class AdamaxTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            ParameterField x = 0.7f;
            ParameterField y = 0.8f;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new Adamax(parameter, alpha: 0.1f));

            List<float> losses = new List<float>(), xs = new List<float>(), ys = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add((float)loss.State);
                xs.Add((float)x.State);
                ys.Add((float)y.State);
            }

            float[] losses_expect = {
                2.017203522e+00f,  1.848746461e+00f,  1.641679806e+00f,  1.380832020e+00f,  1.060975334e+00f,
                7.333544936e-01f,  4.520763172e-01f,  2.451997445e-01f,  1.121434040e-01f,  3.794476879e-02f,
            };

            float[] xs_expect = {
                6.000000009e-01f,  5.218602002e-01f,  4.436312696e-01f,  3.590351748e-01f,  2.747346448e-01f,
                1.964423752e-01f,  1.273338770e-01f,  6.805070112e-02f,  1.772343714e-02f,  -2.504578286e-02f,
            };

            float[] ys_expect = {
                7.000000009e-01f,  6.221711496e-01f,  5.443137712e-01f,  4.602099265e-01f,  3.761952080e-01f,
                2.980234172e-01f,  2.289345075e-01f,  1.696253858e-01f,  1.192595419e-01f,  7.645315564e-02f,
            };

            for (int i = 0; i < loops; i++) {
                Assert.AreEqual(losses_expect[i], losses[i], Math.Abs(losses_expect[i]) * 1e-2f);
                Assert.AreEqual(xs_expect[i], xs[i], Math.Abs(xs[i]) * 1e-2f);
                Assert.AreEqual(ys_expect[i], ys[i], Math.Abs(ys[i]) * 1e-2f);
            }
        }

        [TestMethod]
        public void InitializeTest() {
            const int loops = 10;

            ParameterField x = 0.7f;
            ParameterField y = 0.8f;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new Adamax(parameter, alpha: 0.1f));

            Snapshot snapshot = parameters.Save();

            List<float> losses_first = new List<float>(), xs_first = new List<float>(), ys_first = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_first.Add((float)loss.State);
                xs_first.Add((float)x.State);
                ys_first.Add((float)y.State);
            }

            x.State = 0.7f;
            y.State = 0.8f;

            parameters.InitializeUpdater();

            List<float> losses_second = new List<float>(), xs_second = new List<float>(), ys_second = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_second.Add((float)loss.State);
                xs_second.Add((float)x.State);
                ys_second.Add((float)y.State);
            }

            CollectionAssert.AreEqual(losses_first, losses_second);
            CollectionAssert.AreEqual(xs_first, xs_second);
            CollectionAssert.AreEqual(ys_first, ys_second);

            parameters.Load(snapshot);

            List<float> losses_third = new List<float>(), xs_third = new List<float>(), ys_third = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_third.Add((float)loss.State);
                xs_third.Add((float)x.State);
                ys_third.Add((float)y.State);
            }

            CollectionAssert.AreEqual(losses_first, losses_third);
            CollectionAssert.AreEqual(xs_first, xs_third);
            CollectionAssert.AreEqual(ys_first, ys_third);
        }
    }
}
