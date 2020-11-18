using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class NesterovAGTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            ParameterField x = 0.7f;
            ParameterField y = 0.8f;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new NesterovAG(parameter));

            List<float> losses = new List<float>(), xs = new List<float>(), ys = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add((float)loss.State);
                xs.Add((float)x.State);
                ys.Add((float)y.State);
            }

            float[] losses_expect = {
                2.017204e+00f,  1.978505e+00f,  1.896778e+00f,  1.732103e+00f,  1.435069e+00f,
                1.010019e+00f,  5.824638e-01f,  2.761239e-01f,  9.921303e-02f,  1.705571e-02f,
            };

            float[] xs_expect = {
                6.691874e-01f,  6.211061e-01f,  5.507035e-01f,  4.551741e-01f,  3.411398e-01f,
                2.280485e-01f,  1.318499e-01f,  5.296944e-02f,  -1.384273e-02f,  -7.267004e-02f,
            };

            float[] ys_expect = {
                7.704296e-01f,  7.241055e-01f,  6.560159e-01f,  5.632759e-01f,  4.521616e-01f,
                3.416181e-01f,  2.474374e-01f,  1.702254e-01f,  1.048827e-01f,  4.738320e-02f,
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
            parameters.AddUpdater((parameter) => new NesterovAG(parameter));

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
