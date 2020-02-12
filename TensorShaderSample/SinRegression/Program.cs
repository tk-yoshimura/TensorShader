﻿using System;
using System.Linq;
using TensorShader;
using TensorShaderUtil;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace SinRegression {
    class Program {
        static void Main() {
            const float start = -2, end = +2;
            const int points = 101, loops = 10000;

            Random random = new Random(1234);

            NdimArray<float> xval = NdimArrayUtil.Linspace(start, end, points);
            NdimArray<float> tval = xval.Select((v) => (float)Math.Sin(v));

            VariableField x = (Tensor)xval;
            VariableField t = (Tensor)tval;

            ParameterField p3 = new Tensor(Shape.Scalar, new float[] { (float)random.NextDouble() * 2 - 1 });
            ParameterField p5 = new Tensor(Shape.Scalar, new float[] { (float)random.NextDouble() * 0.02f - 0.01f });
            ParameterField p7 = new Tensor(Shape.Scalar, new float[] { (float)random.NextDouble() * 0.0002f - 0.0001f });

            Field x2 = Square(x);
            Field x3 = x * x2;
            Field x5 = x3 * x2;
            Field x7 = x5 * x2;

            Field y = x + p3 * x3 + p5 * x5 + p7 * x7;

            Field err = AbsoluteError(y, t);

            StoreField sum_err = Sum(err);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            float adam_alpha = 1e-2f;
            parameters.AddUpdater((parameter) => new Adam(parameter, adam_alpha));

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                Console.WriteLine(
                    $"loss:{sum_err.State[0]:E5}, " +
                    $"p3:{p3.State[0]:E5}, " +
                    $"p5:{p5.State[0]:E5}, " +
                    $"p7:{p7.State[0]:E5}");

                parameters["Adam.Alpha"] = adam_alpha *= 0.999f;
            }

            Console.WriteLine("END");
            Console.Read();
        }
    }
}
