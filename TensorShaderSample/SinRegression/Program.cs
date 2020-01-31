using System;
using System.Linq;
using TensorShader;
using TensorShader.Updaters.OptimizeMethod;
using static TensorShader.Field;

namespace SinRegression {
    class Program {
        static void Main() {
            const float start = -2, end = +2;
            const int points = 101, loops = 10000;

            Random random = new Random(1234);

            float[] xval = (new float[points]).Select((_, idx) => start + (end - start) * idx / (points - 1)).ToArray();
            float[] tval = xval.Select((v) => (float)Math.Sin(v)).ToArray();

            VariableField x = new Tensor(Shape.Vector(points), xval);
            VariableField t = new Tensor(Shape.Vector(points), tval);

            ParameterField p3 = new Tensor(Shape.Scalar(), new float[]{ (float)random.NextDouble() * 2 - 1 });
            ParameterField p5 = new Tensor(Shape.Scalar(), new float[]{ (float)random.NextDouble() * 0.02f - 0.01f });
            ParameterField p7 = new Tensor(Shape.Scalar(), new float[]{ (float)random.NextDouble() * 0.0002f - 0.0001f });
            
            Field x2 = Square(x);
            Field x3 = x * x2;
            Field x5 = x3 * x2;
            Field x7 = x5 * x2;

            Field y = x + p3 * x3 + p5 * x5 + p7 * x7;

            Field err = AbsoluteError(y, t);

            StoreField loss = err.Save();

            (Flow flow, Parameters parameters) = Flow.Optimize(err);
            float adam_alpha = 1e-2f;
            parameters.AddUpdater((parameter) => new Adam(parameter, adam_alpha));

            for(int i = 0; i < loops; i++) {
                flow.Execute();
                parameters.Update();

                Console.WriteLine(
                    $"loss:{loss.State[0]:E5}, " +
                    $"p3:{p3.ValueTensor.State[0]:E5}, " +
                    $"p5:{p5.ValueTensor.State[0]:E5}, " + 
                    $"p7:{p7.ValueTensor.State[0]:E5}");

                parameters["Adam.Alpha"] = adam_alpha *= 0.999f;
            }

            Console.WriteLine("END");
            Console.Read();
        }
    }
}
