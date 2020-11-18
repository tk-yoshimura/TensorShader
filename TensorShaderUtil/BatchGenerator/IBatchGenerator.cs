using TensorShader;

namespace TensorShaderUtil.BatchGenerator {
    /// <summary>バッチ生成</summary>
    public interface IBatchGenerator {
        /// <summary>データ形状</summary>
        Shape DataShape { get; }

        /// <summary>バッチ形状</summary>
        Shape BatchShape { get; }

        /// <summary>バッチ数</summary>
        int NumBatches { get; }

        /// <summary>データ生成</summary>
        /// <param name="index">データインデクス</param>
        NdimArray<float> GenerateData(int index);

        /// <summary>データ生成をリクエスト</summary>
        /// <param name="indexes">バッチのデータインデクス</param>
        void Request(int[] indexes = null);

        /// <summary>バッチを受け取る</summary>
        NdimArray<float> Receive();

        /// <summary>バッチを取得する</summary>
        NdimArray<float> Get(int[] indexes = null);
    }
}
