using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShaderUtil.SnapshotSaver;

namespace TensorShaderUtilTest.SnapshotSaver {
    [TestClass]
    public class DefaultBinaryShapshotSaverTest {
        [TestMethod]
        public void ExecuteTest() {
            Snapshot snapshot = new Snapshot();
            snapshot.Append("item1", Shape.Map0D(2, 3), new float[] { 0f, 1f, 2f, 3f, 4f, 5f });
            snapshot.Append("item2", Shape.Map0D(3, 4), new float[] { 5f, 3f, 1f, 0f, 2f, 4f, 1f, 0f, 5f, 3f, 2f, 4f });

            DefaultBinaryShapshotSaver saver = new DefaultBinaryShapshotSaver();

            byte[] data = null;

            using (var stream = new MemoryStream()) {
                saver.Save(stream, snapshot);

                data = stream.ToArray();
            }

            Snapshot snapshot2 = null;
            using (var stream = new MemoryStream(data)) {
                snapshot2 = saver.Load(stream);
            }

            CollectionAssert.AreEquivalent(new string[] { "item1", "item2" }, snapshot2.Keys.ToArray());
            Assert.AreEqual(snapshot.Table["item1"].shape, snapshot2.Table["item1"].shape);
            CollectionAssert.AreEqual(snapshot.Table["item1"].state, snapshot2.Table["item1"].state);
            Assert.AreEqual(snapshot.Table["item2"].shape, snapshot2.Table["item2"].shape);
            CollectionAssert.AreEqual(snapshot.Table["item2"].state, snapshot2.Table["item2"].state);

            saver.Save("debug.tss", snapshot);
            Snapshot snapshot3 = saver.Load("debug.tss");

            CollectionAssert.AreEquivalent(new string[] { "item1", "item2" }, snapshot3.Keys.ToArray());
            Assert.AreEqual(snapshot.Table["item1"].shape, snapshot3.Table["item1"].shape);
            CollectionAssert.AreEqual(snapshot.Table["item1"].state, snapshot3.Table["item1"].state);
            Assert.AreEqual(snapshot.Table["item2"].shape, snapshot3.Table["item2"].shape);
            CollectionAssert.AreEqual(snapshot.Table["item2"].state, snapshot3.Table["item2"].state);

            File.Delete("debug.tss");
        }
    }
}
