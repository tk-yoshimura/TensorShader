using System;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>ソート</summary>
        public static Field Sort(Field x, int axis) {
            Field y = new();
            Link link;

            if (x.EnableBackprop) {
                VariableField index = Tensor.Index(x.Shape, axis);

                link = new Links.ArrayManipulation.SortWithBackprop(x, index, y, axis);
            }
            else {
                link = new Links.ArrayManipulation.SortWithoutBackprop(x, y, axis);
            }

            link.Forward();
            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>ソート</summary>
    public class SortWithBackprop : Link {
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
        public SortWithBackprop(Field x, Field index, Field y, int axis)
            : base(new Field[] { x, index }, y) {

            if (!x.EnableBackprop) {
                throw new ArgumentException($"{nameof(x)}.EnableBackprop");
            }

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
            if (Y.Grad is null) {
                return;
            }

            if (X.EnableBackprop) {
                (_, VariableNode value) = SortWithKey(index_sorted, Y.Grad, Axis);

                X.AddGrad(value);
            }
        }
    }

    /// <summary>ソート</summary>
    public class SortWithoutBackprop : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>インデクス</summary>
        protected Field Index => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>軸</summary>
        protected readonly int Axis;

        /// <summary>コンストラクタ</summary>
        public SortWithoutBackprop(Field x, Field y, int axis)
            : base(new Field[] { x }, y) {

            if (x.EnableBackprop) {
                throw new ArgumentException($"{nameof(x)}.EnableBackprop");
            }

            this.Axis = axis;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            VariableNode y = Sort(X.Value, Axis);

            Y.AssignValue(y);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            return;
        }
    }
}
