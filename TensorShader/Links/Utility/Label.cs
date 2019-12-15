namespace TensorShader {
    public partial class Field {
        /// <summary>ラベル</summary>
        public static Field Label(Field x, string name) {
            Field y = new Field();
            Link link = new Links.Utility.Label(x, y, name);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Utility {
    /// <summary>ラベル</summary>
    public class Label : Link {
        private readonly string name;

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>ラベル名</summary>
        public override string Name => name;

        /// <summary>コンストラクタ</summary>
        public Label(Field infield, Field outfield, string name)
            : base(new Field[] { infield }, outfield) {
            this.name = name;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(X.Value);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y.Grad);
            }
        }

    }
}
