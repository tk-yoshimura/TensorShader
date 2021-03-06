using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using TensorShaderUtil.Iterator;

namespace TensorShaderUtilTest.Iterator {
    [TestClass]
    public class RepeatIteratorTest {
        [TestMethod]
        public void ExecuteTest1() {
            RepeatIterator iterator = new(50, 101);
            RepeatIterator iterator_skip = new(50, 101);

            int[] indexes1 = iterator.Next();
            int[] indexes2 = iterator.Next();
            int[] indexes3 = iterator.Next();
            int[] indexes4 = iterator.Next();
            int[] indexes5 = iterator.Next();
            _ = iterator.Next();
            _ = iterator.Next();
            _ = iterator.Next();

            Assert.AreEqual(50, iterator.NumBatches);
            Assert.AreEqual(101, iterator.Counts);
            Assert.AreEqual(3, iterator.Epoch);
            Assert.AreEqual(8, iterator.Iteration);

            CollectionAssert.AreNotEquivalent(indexes1, indexes2);
            CollectionAssert.AreNotEquivalent(indexes2, indexes3);
            CollectionAssert.AreNotEquivalent(indexes3, indexes4);

            iterator_skip.SkipIteration(1);
            int[] indexes2_skip = iterator_skip.Next();
            Assert.AreEqual(0, iterator_skip.Epoch);
            Assert.AreEqual(2, iterator_skip.Iteration);
            CollectionAssert.AreEquivalent(indexes2, indexes2_skip);

            iterator_skip.SkipEpoch(1);
            int[] indexes5_skip = iterator_skip.Next();
            Assert.AreEqual(2, iterator_skip.Epoch);
            Assert.AreEqual(5, iterator_skip.Iteration);
            CollectionAssert.AreEquivalent(indexes5, indexes5_skip);
        }

        [TestMethod]
        public void ExecuteTest2() {
            RepeatIterator iterator = new(50, 100);
            RepeatIterator iterator_skip = new(50, 100);

            int[] indexes1 = iterator.Next();
            int[] indexes2 = iterator.Next();
            int[] indexes3 = iterator.Next();
            int[] indexes4 = iterator.Next();
            int[] indexes5 = iterator.Next();
            _ = iterator.Next();
            _ = iterator.Next();
            _ = iterator.Next();

            Assert.AreEqual(50, iterator.NumBatches);
            Assert.AreEqual(100, iterator.Counts);
            Assert.AreEqual(4, iterator.Epoch);
            Assert.AreEqual(8, iterator.Iteration);

            CollectionAssert.AreNotEquivalent(indexes1, indexes2);
            CollectionAssert.AreNotEquivalent(indexes2, indexes3);
            CollectionAssert.AreNotEquivalent(indexes3, indexes4);

            iterator_skip.SkipIteration(1);
            int[] indexes2_skip = iterator_skip.Next();
            Assert.AreEqual(1, iterator_skip.Epoch);
            Assert.AreEqual(2, iterator_skip.Iteration);
            CollectionAssert.AreEquivalent(indexes2, indexes2_skip);

            iterator_skip.SkipEpoch(1);
            int[] indexes5_skip = iterator_skip.Next();
            Assert.AreEqual(2, iterator_skip.Epoch);
            Assert.AreEqual(5, iterator_skip.Iteration);
            CollectionAssert.AreEquivalent(indexes5, indexes5_skip);
        }

        [TestMethod]
        public void ExecuteTest3() {
            RepeatIterator iterator = new(1, 4);

            iterator.IncreasedEpoch += (iter) => { Console.WriteLine($"Epoch {iter.Epoch}"); };
            iterator.IncreasedIteration += (iter) => { Console.WriteLine($"Iteration {iter.Iteration}"); };

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
