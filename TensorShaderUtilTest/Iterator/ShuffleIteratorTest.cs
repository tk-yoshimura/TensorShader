using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderUtil.Iterator;

namespace TensorShaderUtilTest.Iterator {
    [TestClass]
    public class ShuffleIteratorTest {
        [TestMethod]
        public void ExecuteTest1() {
            ShuffleIterator iterator = new ShuffleIterator(50, 101, new Random());

            int[] indexes1 = iterator.Next();
            int[] indexes2 = iterator.Next();
            int[] indexes3 = iterator.Next();
            int[] indexes4 = iterator.Next();
            int[] indexes5 = iterator.Next();
            int[] indexes6 = iterator.Next();
            int[] indexes7 = iterator.Next();
            int[] indexes8 = iterator.Next();

            Assert.AreEqual(50, iterator.NumBatches);
            Assert.AreEqual(101, iterator.Counts);
            Assert.AreEqual(3, iterator.Epoch);
            Assert.AreEqual(8, iterator.Iteration);

            CollectionAssert.AreNotEquivalent(indexes1, indexes2);
            CollectionAssert.AreNotEquivalent(indexes3, indexes4);
            CollectionAssert.AreNotEqual(indexes1, indexes3);
        }

        [TestMethod]
        public void ExecuteTest2() {
            ShuffleIterator iterator = new ShuffleIterator(50, 100, new Random());

            int[] indexes1 = iterator.Next();
            int[] indexes2 = iterator.Next();
            int[] indexes3 = iterator.Next();
            int[] indexes4 = iterator.Next();
            int[] indexes5 = iterator.Next();
            int[] indexes6 = iterator.Next();
            int[] indexes7 = iterator.Next();
            int[] indexes8 = iterator.Next();

            Assert.AreEqual(50, iterator.NumBatches);
            Assert.AreEqual(100, iterator.Counts);
            Assert.AreEqual(4, iterator.Epoch);
            Assert.AreEqual(8, iterator.Iteration);

            CollectionAssert.AreNotEquivalent(indexes1, indexes2);
            CollectionAssert.AreNotEquivalent(indexes3, indexes4);
            CollectionAssert.AreNotEqual(indexes1, indexes3);
        }

        [TestMethod]
        public void ExecuteTest3() {
            ShuffleIterator iterator = new ShuffleIterator(1, 4, new Random());

            iterator.IncreasedEpoch += (ite, e) => { Console.WriteLine($"Epoch {ite.Epoch}"); };
            iterator.IncreasedIteration += (ite, e) => { Console.WriteLine($"Iteration {ite.Iteration}"); };

            int[] indexes1 = iterator.Next();
            int[] indexes2 = iterator.Next();
            int[] indexes3 = iterator.Next();
            int[] indexes4 = iterator.Next();
            int[] indexes5 = iterator.Next();
            int[] indexes6 = iterator.Next();
            int[] indexes7 = iterator.Next();
            int[] indexes8 = iterator.Next();

            Assert.AreEqual(1, iterator.NumBatches);
            Assert.AreEqual(4, iterator.Counts);
            Assert.AreEqual(2, iterator.Epoch);
            Assert.AreEqual(8, iterator.Iteration);
        }
    }
}
