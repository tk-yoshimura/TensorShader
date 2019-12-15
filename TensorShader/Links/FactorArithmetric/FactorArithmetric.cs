namespace TensorShader.Links.FactorArithmetric {
    /// <summary>指数付きテンソル1項演算</summary>
    public abstract class FactorArithmetric : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>指数項</summary>
        protected Field Factor => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public FactorArithmetric(Field infield, Field factorfield, Field outfield)
            : base(new Field[] { infield, factorfield }, outfield) { }
    }
}
