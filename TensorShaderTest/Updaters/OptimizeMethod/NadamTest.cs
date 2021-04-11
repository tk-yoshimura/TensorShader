using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class NadamTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            ParameterField x = 0.7f;
            ParameterField y = 0.8f;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new Nadam(parameter, alpha: 0.1f));

            List<float> losses = new(), xs = new(), ys = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add(loss.State);
                xs.Add(x.State);
                ys.Add(y.State);
            }

            float[] losses_expect = {
                2.017203522e+00f,  1.604613870e+00f,  1.048012517e+00f,  5.676715677e-01f,  2.518899043e-01f,
                8.113055659e-02f,  9.838303339e-03f,  4.401829741e-03f,  4.300932598e-02f,  1.088092727e-01f,
            };

            float[] xs_expect = {
                5.100000007e-01f,  3.563114663e-01f,  2.307797325e-01f,  1.309010908e-01f,  5.108071149e-02f,
                -1.512194313e-02f,  -7.220467539e-02f,  -1.224719240e-01f,  -1.664595196e-01f,  -2.034850690e-01f,
            };

            float[] ys_expect = {
                6.100000008e-01f,  4.562977857e-01f,  3.305791975e-01f,  2.304412294e-01f,  1.504041878e-01f,
                8.405014125e-02f,  2.685437093e-02f,  -2.351134319e-02f,  -6.758124197e-02f,  -1.046422414e-01f,
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
            parameters.AddUpdater((parameter) => new Nadam(parameter, alpha: 0.1f));

            Snapshot snapshot = parameters.Save();

            List<float> losses_first = new(), xs_first = new(), ys_first = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_first.Add(loss.State);
                xs_first.Add(x.State);
                ys_first.Add(y.State);
            }

            x.State = 0.7f;
            y.State = 0.8f;

            parameters.InitializeUpdater();

            List<float> losses_second = new(), xs_second = new(), ys_second = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_second.Add(loss.State);
                xs_second.Add(x.State);
                ys_second.Add(y.State);
            }

            CollectionAssert.AreEqual(losses_first, losses_second);
            CollectionAssert.AreEqual(xs_first, xs_second);
            CollectionAssert.AreEqual(ys_first, ys_second);

            parameters.Load(snapshot);

            List<float> losses_third = new(), xs_third = new(), ys_third = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_third.Add(loss.State);
                xs_third.Add(x.State);
                ys_third.Add(y.State);
            }

            CollectionAssert.AreEqual(losses_first, losses_third);
            CollectionAssert.AreEqual(xs_first, xs_third);
            CollectionAssert.AreEqual(ys_first, ys_third);
        }
    }
}
