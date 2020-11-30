using System;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace SearchMinimum {
    class Program {
        static void Main() {
            const int loops = 1000;
            const float x_init = 4, y_init = 3;

            ParameterField x = x_init;
            ParameterField y = y_init;
            VariableField r = 0.05f;

            // f(x, y) = x^2 + y^2
            Field f = Square(x) + Square(y);
            // g(x, y) = sin(x + sin(y))^2 + sin(y + sin(x))^2
            Field g = Square(Sin(x + Sin(y))) + Square(Sin(y + Sin(x)));
            // h(x, y) = f(x, y) * r + g(x, y)
            StoreField h = f * r + g;

            (Flow flow, Parameters parameters) = Flow.Optimize(h);
            parameters.AddUpdater((parameter) => new Adam(parameter, alpha: 0.1f));

            for (int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                Console.WriteLine($"h(x, y):{(float)h.State:E5}, x:{(float)x.State:E5}, y:{(float)y.State:E5}");
            }

            Console.WriteLine("END");
            Console.Read();
        }

        /* in maxima
            > f(x, y) := x^2 + y^2;
            > g(x, y) := sin(x + sin(y))^2 + sin(y + sin(x))^2;
            > h(x, y) := f(x, y) / 20 + g(x, y);
            > plot3d(h(x, y), [x,-5,5], [y,-5,5]);
        */
    }
}
