using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderUtil.ScheduleUtil;

namespace TensorShaderUtilTest.ScheduleUtil {
    [TestClass]
    public class ConvergenceTesterTest {
        [TestMethod]
        public void ExecuteTest() {
            Random random = new Random(1234);

            ConvergenceTester tester = new ConvergenceTester(32, 0.5);

            for (int rs = 0; rs <= 100; rs += 10) {
                Console.WriteLine($"rs: {rs}");

                for (int i = 0; i < 64; i++) {
                    tester.Next(i + rs * random.NextDouble());

                    Console.WriteLine($"i: {i}, R:{tester.R()}, IsConvergenced: {tester.IsConvergenced}");
                }

                tester.Initialize();
            }

            for (int d = -4; d <= 4; d++) {
                Console.WriteLine($"d: {d}");

                for (int i = 0; i < 64; i++) {
                    tester.Next(d * i + 100 * random.NextDouble());

                    Console.WriteLine($"i: {i}, R:{tester.R()}, IsConvergenced: {tester.IsConvergenced}");
                }

                tester.Initialize();
            }
        }

        [TestMethod]
        public void TheoreticalTest() {
            ConvergenceTester tester = new ConvergenceTester(32, 0.5);

            for (int i = 0; i < 64; i++) {
                tester.Next(i);

                if (i >= tester.Length) {
                    Assert.AreEqual(1.0, tester.R());
                }
            }

            for (int i = 0; i < 64; i++) {
                tester.Next(0);

                if (i >= tester.Length) {
                    Assert.AreEqual(0, tester.R());
                }
            }

            for (int i = 0; i < 64; i++) {
                tester.Next(-i);

                if (i >= tester.Length) {
                    Assert.AreEqual(-1.0, tester.R());
                }
            }
        }
    }
}
