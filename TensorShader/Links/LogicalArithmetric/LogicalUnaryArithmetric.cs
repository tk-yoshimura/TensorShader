namespace TensorShader.Links.LogicalArithmetric {
    /// <summary>論理演算</summary>
    public abstract class LogicalUnaryArithmetric : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public LogicalUnaryArithmetric(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) { }

        /// <summary>逆伝搬</summary>
        public override sealed void Backward() {
            return;
        }
    }
}
