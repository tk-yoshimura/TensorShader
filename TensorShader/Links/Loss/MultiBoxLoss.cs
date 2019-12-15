using System;

namespace TensorShader {
    public partial class Field {
        /// <summary>マルチボックス損失</summary>
        /// <remarks>
        /// Wei Liu, Dragomir Anguelov, Dumitru Erhan.
        /// Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
        /// SSD: Single Shot MultiBox Detector. ECCV 2016.
        /// </remarks>
        /// <param name="loc">検出位置 shape:(E, B, N) E:尺度要素数 B:ボックス数 N:バッチ数</param>
        /// <param name="gt">正解位置 shape:(E, B, N) E:尺度要素数 B:ボックス数 N:バッチ数</param>
        /// <param name="conf">識別結果 shape:(C, B, N) C:クラス数 B:ボックス数 N:バッチ数</param>
        /// <param name="label">正解クラス shape:(B, N) B:ボックス数 N:バッチ数</param>
        /// <param name="hard_negative_mining_rate">HardNegativeMiningにおいて背景ボックス数を物体ボックス数の何倍採用するか</param>
        /// <returns>損失 shape(B, N)</returns>
        public static (Field locloss, Field confloss) MultiBoxLoss(Field loc, Field gt, Field conf, Field label, float hard_negative_mining_rate = 3) {
            if (loc.Shape.Ndim != 3 || loc.Shape != gt.Shape) {
                throw new ArgumentException($"{nameof(loc)}, {nameof(gt)}");
            }

            if (conf.Shape.Ndim != 3 || label.Shape.Ndim != 2
                || loc.Shape.Width != label.Shape.Channels || loc.Shape.Batch != label.Shape.Batch
                || conf.Shape.Width != label.Shape.Channels || conf.Shape.Batch != label.Shape.Batch) {
                throw new ArgumentException($"{nameof(conf)}, {nameof(label)}");
            }

            int boxes_axis = 0, classes = conf.Shape.Channels;

            Field positives = GreaterThan(label, 0.5f), negatives = positives - 1;

            Field locloss_huber = Sum(HuberLoss(loc, gt, delta: 1), axes: new int[] { Axis.Map0D.Channels });
            Field locloss = locloss_huber * positives;

            Field confloss_softmax = Sum(SoftmaxCrossEntropy(conf, OneHotVector(label, classes)), axes: new int[] { Axis.Map0D.Channels });
            Field negatives_sort = ArgSort(ArgSort(confloss_softmax * negatives, boxes_axis), boxes_axis);
            Field negatives_minings = hard_negative_mining_rate * Sum(positives, axes: new int[] { boxes_axis }, keepdims: true);
            Field negatives_hard = LessThan(negatives_sort, Broadcast(negatives_minings, negatives_sort.Shape));
            Field confloss = confloss_softmax * Or(positives, negatives_hard);

            return (locloss, confloss);
        }
    }
}
