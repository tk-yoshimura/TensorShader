using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ソート</summary>
        /// <remarks>安定ソート</remarks>
        public static Field ArgSort(Field x, int axis) {
            Field y = new();
            VariableField index = Tensor.Index(x.Shape, axis);

            Link link = new Links.ArrayManipulation.ArgSort(x, index, y, axis);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>ソート</summary>
    public class ArgSort : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>インデクス</summary>
        protected Field Index => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>軸</summary>
        protected readonly int Axis;

        /// <summary>コンストラクタ</summary>
        public ArgSort(Field x, Field index, Field y, int axis)
            : base(new Field[] { x, index }, y) {
            this.Axis = axis;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            (_, VariableNode value) = SortWithKey(X.Value, Index.Value, Axis);

            Y.AssignValue(value);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            return;
        }
    }
}
