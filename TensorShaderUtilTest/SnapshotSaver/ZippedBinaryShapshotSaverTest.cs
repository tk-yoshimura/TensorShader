using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Linq;
using TensorShader;
using TensorShaderUtil.SnapshotSaver;

namespace TensorShaderUtilTest.SnapshotSaver {
    [TestClass]
    public class ZippedBinaryShapshotSaverTest {
        [TestMethod]
        public void ExecuteTest() {
            Snapshot snapshot = new();
            snapshot.Append("item0", Shape.Vector(5), new float[] { 1f, 2f, 3f, 4f, 5f });
            snapshot.Append("item1", Shape.Map0D(2, 3), new float[] { 0f, 1f, 2f, 3f, 4f, 5f });
            snapshot.Append("item2", Shape.Map0D(3, 4), new float[] { 5f, 3f, 1f, 0f, 2f, 4f, 1f, 0f, 5f, 3f, 2f, 4f });
            snapshot.Append("item3", Shape.Scalar, new float[] { 2f });
            snapshot.Append("item4", Shape.Map1D(2, 1, 4), new float[] { 3f, 1f, 0f, 2f, 4f, 1f, 0f, 5f });

            string[] items = new string[] { "item0", "item1", "item2", "item3", "item4" };

            ZippedBinaryShapshotSaver saver = new();

            byte[] data = null;

            using (var stream = new MemoryStream()) {
                saver.Save(stream, snapshot);

                data = stream.ToArray();
            }

            Snapshot snapshot2 = null;
            using (var stream = new MemoryStream(data)) {
                snapshot2 = saver.Load(stream);
            }

            CollectionAssert.AreEquivalent(items, snapshot2.Keys.ToArray());

            foreach (string item in items) {
                Assert.AreEqual(snapshot.Table[item].Shape, snapshot2.Table[item].Shape);
                CollectionAssert.AreEqual(snapshot.Table[item].Value, snapshot2.Table[item].Value);
            }

            saver.Save("debug.tss", snapshot);
            Snapshot snapshot3 = saver.Load("debug.tss");

            CollectionAssert.AreEquivalent(items, snapshot3.Keys.ToArray());

            foreach (string item in items) {
                Assert.AreEqual(snapshot.Table[item].Shape, snapshot3.Table[item].Shape);
                CollectionAssert.AreEqual(snapshot.Table[item].Value, snapshot3.Table[item].Value);
            }

            File.Delete("debug.tss");
        }
    }
}
