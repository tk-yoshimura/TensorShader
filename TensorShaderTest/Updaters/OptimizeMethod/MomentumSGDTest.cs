using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class MomentumSGDTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            ParameterField x = 0.7f;
            ParameterField y = 0.8f;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            StoreField loss = f / 20 + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new MomentumSGD(parameter, lambda: 0.01f, alpha: 0.9f));

            List<float> losses = new(), xs = new(), ys = new();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add(loss.State);
                xs.Add(x.State);
                ys.Add(y.State);
            }

            float[] losses_expect = {
                2.017204e+00f,  2.004166e+00f,  1.973552e+00f,  1.914177e+00f,  1.806149e+00f,
                1.620532e+00f,  1.329737e+00f,  9.403230e-01f,  5.278062e-01f,  2.037859e-01f,
            };

            float[] xs_expect = {
                6.886300e-01f,  6.657792e-01f,  6.301770e-01f,  5.796571e-01f,  5.117803e-01f,
                4.253171e-01f,  3.227444e-01f,  2.118561e-01f,  1.027868e-01f,  2.160184e-03f,
            };

            float[] ys_expect = {
                7.890884e-01f,  7.671345e-01f,  7.328712e-01f,  6.841461e-01f,  6.185205e-01f,
                5.347088e-01f,  4.350275e-01f,  3.270243e-01f,  2.206320e-01f,  1.224137e-01f,
            };

            for (int i = 0; i < loops; i++) {
                Assert.AreEqual(losses_expect[i], losses[i], Math.Abs(losses_expect[i]) * 1e-4f);
                Assert.AreEqual(xs_expect[i], xs[i], Math.Abs(xs[i]) * 1e-4f);
                Assert.AreEqual(ys_expect[i], ys[i], Math.Abs(ys[i]) * 1e-4f);
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
            parameters.AddUpdater((parameter) => new MomentumSGD(parameter, lambda: 0.01f, alpha: 0.9f));

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
