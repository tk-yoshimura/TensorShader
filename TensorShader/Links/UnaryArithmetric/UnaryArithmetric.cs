namespace TensorShader.Links.UnaryArithmetric {
    /// <summary>1項演算</summary>
    public abstract class UnaryArithmetric : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public UnaryArithmetric(Field infield, Field outfield)
            : base(new Field[] { infield }, outfield) { }

    }
}
