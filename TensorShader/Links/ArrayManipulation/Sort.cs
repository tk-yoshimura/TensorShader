using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ソート</summary>
        public static Field Sort(Field x, int axis) {
            Field y = new Field();
            VariableField index = Tensor.Index(x.Shape, axis);

            Link link = new Links.ArrayManipulation.Sort(x, index, y, axis);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>ソート</summary>
    public class Sort : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>インデクス</summary>
        protected Field Index => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>軸</summary>
        protected readonly int Axis;

        private VariableNode index_sorted;

        /// <summary>コンストラクタ</summary>
        public Sort(Field x, Field index, Field y, int axis)
            : base(new Field[] { x, index }, y) {
            this.Axis = axis;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            (VariableNode key, VariableNode value) = SortWithKey(X.Value, Index.Value, Axis);

            this.index_sorted = value;

            Y.AssignValue(key);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            return;
        }
    }
}
