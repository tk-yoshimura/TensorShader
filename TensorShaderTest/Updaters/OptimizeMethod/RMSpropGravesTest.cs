using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace TensorShaderTest.Updaters.OptimizeMethod {
    [TestClass]
    public class RMSpropGravesTest {
        [TestMethod]
        public void ReferenceTest() {
            const int loops = 10;

            Tensor x_tensor = new Tensor(Shape.Scalar(), new float[] { 0.7f });
            Tensor y_tensor = new Tensor(Shape.Scalar(), new float[] { 0.8f });

            ParameterField x = x_tensor, y = y_tensor;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            Field loss = f / 20 + g;
            StoreField lossnode = loss;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new RMSpropGraves(parameter, lambda: 0.01f, rho: 0.9f));

            List<float> losses = new List<float>(), xs = new List<float>(), ys = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses.Add(lossnode.State[0]);
                xs.Add(x_tensor.State[0]);
                ys.Add(y_tensor.State[0]);
            }

            float[] losses_expect = {
                2.017204e+00f,  1.973906e+00f,  1.926629e+00f,  1.874735e+00f,  1.818124e+00f,
                1.756843e+00f,  1.691031e+00f,  1.620906e+00f,  1.546772e+00f,  1.469024e+00f,
            };

            float[] xs_expect = {
                6.666810e-01f,  6.382151e-01f,  6.119421e-01f,  5.869045e-01f,  5.626309e-01f,
                5.388539e-01f,  5.154108e-01f,  4.922010e-01f,  4.691655e-01f,  4.462774e-01f,
            };

            float[] ys_expect = {
                7.666822e-01f,  7.381649e-01f,  7.118302e-01f,  6.867288e-01f,  6.623917e-01f,
                6.385518e-01f,  6.150461e-01f,  5.917734e-01f,  5.686740e-01f,  5.457199e-01f,
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

            Tensor x_tensor = new Tensor(Shape.Scalar(), new float[] { 0.7f });
            Tensor y_tensor = new Tensor(Shape.Scalar(), new float[] { 0.8f });

            ParameterField x = x_tensor, y = y_tensor;

            Field f = Square(x) + Square(y);
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));

            Field loss = f / 20 + g;
            StoreField lossnode = loss;

            (Flow flow, Parameters parameters) = Flow.Optimize(loss);
            parameters.AddUpdater((parameter) => new RMSpropGraves(parameter, lambda: 0.1f, rho: 0.9f));

            Snapshot snapshot = parameters.Save();

            List<float> losses_first = new List<float>(), xs_first = new List<float>(), ys_first = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_first.Add(lossnode.State[0]);
                xs_first.Add(x_tensor.State[0]);
                ys_first.Add(y_tensor.State[0]);
            }

            x_tensor.State = new float[] { 0.7f };
            y_tensor.State = new float[] { 0.8f };

            parameters.InitializeUpdater();

            List<float> losses_second = new List<float>(), xs_second = new List<float>(), ys_second = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_second.Add(lossnode.State[0]);
                xs_second.Add(x_tensor.State[0]);
                ys_second.Add(y_tensor.State[0]);
            }

            CollectionAssert.AreEqual(losses_first, losses_second);
            CollectionAssert.AreEqual(xs_first, xs_second);
            CollectionAssert.AreEqual(ys_first, ys_second);

            parameters.Load(snapshot);

            List<float> losses_third = new List<float>(), xs_third = new List<float>(), ys_third = new List<float>();

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                losses_third.Add(lossnode.State[0]);
                xs_third.Add(x_tensor.State[0]);
                ys_third.Add(y_tensor.State[0]);
            }

            CollectionAssert.AreEqual(losses_first, losses_third);
            CollectionAssert.AreEqual(xs_first, xs_third);
            CollectionAssert.AreEqual(ys_first, ys_third);
        }
    }
}
