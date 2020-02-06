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

            Random random = new Random(1234);

            float[] xval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();
            float[] yval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();
            float[] tval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();

            float[] sin_xy_expected, dx_expected, dy_expected;
            float[] sin_xy_actual, dx_actual, dy_actual;

            /*expected*/
            {
                ParameterField x = new Tensor(Shape.Vector(points), xval);
                ParameterField y = new Tensor(Shape.Vector(points), yval);
                VariableField t = new Tensor(Shape.Vector(points), tval);

                Field sin_xy = Sin(x * y);
                Field err = sin_xy - t;
                StoreField sin_xy_store = sin_xy.Save();

                (Flow flow, _) = Flow.Optimize(err);

                flow.Execute();

                sin_xy_expected = sin_xy_store.State;
                dx_expected = x.GradTensor.State;
                dy_expected = y.GradTensor.State;
            }

            /*actual*/
            {
                ParameterField x = new Tensor(Shape.Vector(points), xval);
                ParameterField y = new Tensor(Shape.Vector(points), yval);
                VariableField t = new Tensor(Shape.Vector(points), tval);

                Field sin_xy = CustomBinaryArithmetric.SinXY(x, y);
                Field err = sin_xy - t;
                StoreField sin_xy_store = sin_xy.Save();

                (Flow flow, _) = Flow.Optimize(err);

                flow.Execute();

                sin_xy_actual = sin_xy_store.State;
                dx_actual = x.GradTensor.State;
                dy_actual = y.GradTensor.State;
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

            Random random = new Random(1234);

            float[] xval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();
            float[] tval = (new float[points]).Select((_) => (float)(random.NextDouble() * 2 - 1)).ToArray();

            float[] expsin_expected, dx_expected;
            float[] expsin_actual, dx_actual;

            /*expected*/
            {
                ParameterField x = new Tensor(Shape.Vector(points), xval);
                VariableField t = new Tensor(Shape.Vector(points), tval);

                Field expsin = Exp(Sin(x));
                Field err = expsin - t;
                StoreField expsin_store = expsin.Save();

                (Flow flow, _) = Flow.Optimize(err);

                flow.Execute();

                expsin_expected = expsin_store.State;
                dx_expected = x.GradTensor.State;
            }

            /*actual*/
            {
                ParameterField x = new Tensor(Shape.Vector(points), xval);
                VariableField t = new Tensor(Shape.Vector(points), tval);

                Field expsin = CustomUnaryArithmetric.ExpSin(x);
                Field err = expsin - t;
                StoreField expsin_store = expsin.Save();

                (Flow flow, _) = Flow.Optimize(err);

                flow.Execute();

                expsin_actual = expsin_store.State;
                dx_actual = x.GradTensor.State;
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
