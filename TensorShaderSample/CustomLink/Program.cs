using System;
using System.Linq;
using TensorShader;

using static TensorShader.Field;

namespace CustomLink {
    class Program {
        static void Main() {
            TestSinXY();
            TestExpSin();

            Console.WriteLine("END");
            Console.Read();
        }

        static void TestSinXY() {
            const int points = 32;

            Random random = new(1234);

            float[] xval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();
            float[] yval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();
            float[] tval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();

            float[] sin_xy_expected, dx_expected, dy_expected;
            float[] sin_xy_actual, dx_actual, dy_actual;

            /*expected*/
            {
                ParameterField x = xval;
                ParameterField y = yval;
                VariableField t = tval;

                StoreField sin_xy = Sin(x * y);
                Field err = sin_xy - t;

                (Flow flow, _) = Flow.Optimize(err);

                flow.Execute();

                sin_xy_expected = sin_xy.State.Value;
                dx_expected = x.GradState.Value;
                dy_expected = y.GradState.Value;
            }

            /*actual*/
            {
                ParameterField x = xval;
                ParameterField y = yval;
                VariableField t = tval;

                StoreField sin_xy = CustomBinaryArithmetric.SinXY(x, y);
                Field err = sin_xy - t;

                (Flow flow, _) = Flow.Optimize(err);

                flow.Execute();

                sin_xy_actual = sin_xy.State.Value;
                dx_actual = x.GradState.Value;
                dy_actual = y.GradState.Value;
            }

            Console.WriteLine("x, y, sin_xy(expected), sin_xy(actual), dx(expected), dx(actual), dy(expected), dy(actual)");
            float sin_xy_error = 0, dx_error = 0, dy_error = 0;

            for (int i = 0; i < points; i++) {
                Console.WriteLine(
                    $"{xval[i]}, {yval[i]}, " +
                    $"{sin_xy_expected[i]}, {sin_xy_actual[i]}, " +
                    $"{dx_expected[i]}, {dx_actual[i]}, " +
                    $"{dy_expected[i]}, {dy_actual[i]}");

                sin_xy_error += Math.Abs(sin_xy_expected[i] - sin_xy_actual[i]);
                dx_error += Math.Abs(dx_expected[i] - dx_actual[i]);
                dy_error += Math.Abs(dy_expected[i] - dy_actual[i]);

            }

            Console.WriteLine($"sin_xy(sum error) : {sin_xy_error}");
            Console.WriteLine($"dx_error(sum error) : {dx_error}");
            Console.WriteLine($"dy_error(sum error) : {dy_error}");
        }

        static void TestExpSin() {
            const int points = 32;

            Random random = new(1234);

            float[] xval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();
            float[] tval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();

            float[] expsin_expected, dx_expected;
            float[] expsin_actual, dx_actual;

            /*expected*/
            {
                ParameterField x = xval;
                VariableField t = tval;

                StoreField expsin = Exp(Sin(x));
                Field err = expsin - t;

                (Flow flow, _) = Flow.Optimize(err);

                flow.Execute();

                expsin_expected = expsin.State.Value;
                dx_expected = x.GradState.Value;
            }

            /*actual*/
            {
                ParameterField x = xval;
                VariableField t = tval;

                StoreField expsin = CustomUnaryArithmetric.ExpSin(x);
                Field err = expsin - t;

                (Flow flow, _) = Flow.Optimize(err);

                flow.Execute();

                expsin_actual = expsin.State.Value;
                dx_actual = x.GradState.Value;
            }

            Console.WriteLine("x, expsin(expected), expsin(actual), dx(expected), dx(actual), dy(expected), dy(actual)");
            float expsin_error = 0, dx_error = 0;

            for (int i = 0; i < points; i++) {
                Console.WriteLine(
                    $"{xval[i]}, " +
                    $"{expsin_expected[i]}, {expsin_actual[i]}, " +
                    $"{dx_expected[i]}, {dx_actual[i]}");

                expsin_error += Math.Abs(expsin_expected[i] - expsin_actual[i]);
                dx_error += Math.Abs(dx_expected[i] - dx_actual[i]);
            }

            Console.WriteLine($"expsin(sum error) : {expsin_error}");
            Console.WriteLine($"dx_error(sum error) : {dx_error}");

            Console.WriteLine("END");
            Console.Read();
        }
    }
}
