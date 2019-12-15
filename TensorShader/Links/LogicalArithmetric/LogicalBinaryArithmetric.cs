namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>論理演算</summary>
    public abstract class LogicalBinaryArithmetric : Link {
        /// <summary>入力項1</summary>
        protected Field X1 => InFields[0];

        /// <summary>入力項2</summary>
        protected Field X2 => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public LogicalBinaryArithmetric(Field infield1, Field infield2, Field outfield)
            : base(new Field[] { infield1, infield2 }, outfield) { }

        /// <summary>逆伝搬</summary>
        public override sealed void Backward() {
            return;
        }
    }
}
