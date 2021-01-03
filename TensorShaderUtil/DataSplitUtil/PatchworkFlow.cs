using System;
using System.Linq;
using System.Collections.Generic;
using TensorShader;

namespace TensorShaderUtil.DataSplitUtil {

    /// <summary>イテレータイベント</summary>
    public delegate void ProgressEventHandler(int progress, int all);

    /// <summary>任意サイズのマップに対し計算フローを実行</summary>
    public class PatchworkFlow {

        private readonly Flow flow;
        private readonly VariableField input;
        private readonly StoreField output;
        private readonly int ndim;
        private readonly IReadOnlyList<int> inmapshape, outmapshape, margin;

        /// <summary>ブロック計算完了時イベント</summary>
        public event ProgressEventHandler ProgressEvent;
        
        /// <summary>コンストラクタ</summary>
        /// <param name="flow">計算フロー</param>
        /// <param name="input">入力フィールド</param>
        /// <param name="output">出力フィールド</param>
        /// <param name="margin">出力フィールドへのブロック展開時に切り捨てられるマージン</param>
        /// <remarks>入出力フィールドのマップサイズは整数比である必要がある</remarks>
        public PatchworkFlow(Flow flow, VariableField input, StoreField output, int[] margin) {
            if (margin == null || margin.Length < 1 || margin.Any((m) => m < 0)) {
                throw new ArgumentException(nameof(margin));
            }
            if (input.Shape.Type != ShapeType.Map || output.Shape.Type != ShapeType.Map) {
                throw new ArgumentException(ExceptionMessage.Argument("Type", ShapeType.Map));
            }
            if (input.Shape.Ndim < 3) {
                throw new ArgumentException(ExceptionMessage.Argument("Ndim", ">=3"));
            }
            if (input.Shape.Ndim > 5) {
                throw new ArgumentException(ExceptionMessage.Argument("Ndim", "<=5"));
            }
            if (input.Shape.Ndim != output.Shape.Ndim) {
                throw new ArgumentException(ExceptionMessage.Argument("Ndim", output.Shape.Ndim, input.Shape.Ndim));
            }

            int ndim = input.Shape.Ndim - 2;
            if (margin.Length != ndim) {
                throw new ArgumentException(ExceptionMessage.Argument($"{nameof(margin)}.Length", ndim));
            }

            int[] inmapshape = ((int[])input.Shape).Skip(1).Take(ndim).ToArray();
            int[] outmapshape = ((int[])output.Shape).Skip(1).Take(ndim).ToArray();

            for (int i = 0; i < ndim; i++) {
                if (inmapshape[i] > outmapshape[i]) {
                    if (inmapshape[i] % outmapshape[i] != 0) {
                        throw new ArgumentException(ExceptionMessage.ArgumentMultiple($"{nameof(input)}.Shape[{i + 1}]", inmapshape[i], outmapshape[i]));
                    }
                }
                else {
                    if (outmapshape[i] % inmapshape[i] != 0) {
                        throw new ArgumentException(ExceptionMessage.ArgumentMultiple($"{nameof(output)}.Shape[{i + 1}]", outmapshape[i], inmapshape[i]));
                    }
                }
            }

            for (int i = 0; i < ndim; i++) {
                if (inmapshape[i] > outmapshape[i]) {
                    int s = inmapshape[i] / outmapshape[i];
                    margin[i] = (margin[i] + s - 1) / s * s;
                }

                if (inmapshape[i] <= margin[i] * 2) { 
                    throw new ArgumentException(nameof(margin));
                }
            }

            this.flow = flow;
            this.input = input;
            this.output = output;
            this.ndim = ndim;
            this.inmapshape = inmapshape;
            this.outmapshape = outmapshape;
            this.margin = margin;
        }

        /// <summary>コンストラクタ</summary>
        /// <param name="flow">計算フロー</param>
        /// <param name="input">入力フィールド</param>
        /// <param name="output">出力フィールド</param>
        /// <param name="margin">出力フィールドへのブロック展開時に切り捨てられるマージン</param>
        /// <remarks>入出力フィールドのマップサイズは整数比である必要がある</remarks>
        public PatchworkFlow(Flow flow, VariableField input, StoreField output, int margin)
            : this(flow, input, output, Enumerable.Repeat(margin, input.Shape.Ndim - 2).ToArray()) { }

        /// <summary>計算実行</summary>
        /// <param name="inmap">入力マップ</param>
        /// <returns>出力マップ</returns>
        public NdimArray<float> Execute(NdimArray<float> inmap) {
            if (inmap.Ndim != ndim + 2) {
                throw new ArgumentException(ExceptionMessage.ShapeElements(inmap.Shape, ("Ndim", ndim + 2), ("Type", ShapeType.Map)));
            }

            PatchworkCoord[] incoords = new PatchworkCoord[ndim];
            bool enable_pad = false;
            (int pa, int pb)[] pads = new (int, int)[ndim];

            for (int i = 0; i < incoords.Length; i++) {
                int s = inmap.Shape[i + 1], r = inmapshape[i] / outmapshape[i];

                if (s < inmapshape[i]) {
                    pads[i] = (0, inmapshape[i] - s);
                    enable_pad = true;
                }

                if ((r > 1) && (s % r != 0)) { 
                    s -= s % r;
                }

                incoords[i] = new PatchworkCoord(Math.Max(s, inmapshape[i]), inmapshape[i], margin[i]);
            }

            if (enable_pad) {
                inmap = NdimArray<float>.EdgePaddingND(inmap, pads);
            }

            Shape shape = new Shape(
                ShapeType.Map,
                    (new int[] { inmap.Channels }).Concat(
                    ((int[])inmap.Shape).Skip(1).Take(ndim).Select(
                        (s, i) => RemapCoord(s, inmapshape[i], outmapshape[i]))
                    ).Concat(
                    new int[] { inmap.Batch }
                    ).ToArray()
                );

            NdimArray<float> outmap = shape;

            if (ndim == 1) {
                Execute1D(inmap, outmap, incoords[0]);
            }
            else if (ndim == 2) {
                Execute2D(inmap, outmap, incoords[0], incoords[1]);
            }
            else {
                throw new ArgumentException(nameof(ndim));
            }

            if (enable_pad) {
                (int, int)[] trims = pads.Select(
                    (p, i) => (RemapCoord(p.pa, inmapshape[i], outmapshape[i]), RemapCoord(p.pb, inmapshape[i], outmapshape[i]))
                ).ToArray();

                outmap = NdimArray<float>.TrimmingND(outmap, trims);
            }

            return outmap;
        }

        /// <summary>計算実行(1D)</summary>
        private void Execute1D(NdimArray<float> inmap, NdimArray<float> outmap, PatchworkCoord inxcoords) {
            int block_iw = inxcoords.BlockSize;

            int n = inxcoords.Blocks;

            for (int i = 0, x = 0; x < inxcoords.Blocks; x++, i++) { 
                int patch_ix = inxcoords.PatchCoords[x], patch_iw = inxcoords.PatchSizes[x], block_ix = inxcoords.BlockCoords[x];
                int patch_ox = RemapCoord(patch_ix, inmap.Width, outmap.Width);
                int patch_ow = RemapCoord(patch_iw, inmap.Width, outmap.Width);
                int block_ox = RemapCoord(block_ix, inmap.Width, outmap.Width);

                NdimArray<float> inblock = NdimArray<float>.Slice1D(inmap, block_ix, block_iw);
                input.State = inblock;
                flow.Execute();
                NdimArray<float> outblock = output.State;

                NdimArray<float>.RegionCopy1D(outblock, outmap, 
                    patch_ox - block_ox, patch_ox, patch_ow);

                ProgressEvent?.Invoke(i + 1, n);
            }
        }

        /// <summary>計算実行(2D)</summary>
        private void Execute2D(NdimArray<float> inmap, NdimArray<float> outmap, PatchworkCoord inxcoords, PatchworkCoord inycoords) {
            int block_ih = inycoords.BlockSize, block_iw = inxcoords.BlockSize;

            int n = inycoords.Blocks * inxcoords.Blocks;

            for (int i = 0, y = 0; y < inycoords.Blocks; y++) {
                int patch_iy = inycoords.PatchCoords[y], patch_ih = inycoords.PatchSizes[y], block_iy = inycoords.BlockCoords[y];
                int patch_oy = RemapCoord(patch_iy, inmap.Height, outmap.Height);
                int patch_oh = RemapCoord(patch_ih, inmap.Height, outmap.Height);
                int block_oy = RemapCoord(block_iy, inmap.Height, outmap.Height);

                for (int x = 0; x < inxcoords.Blocks; x++, i++) { 
                    int patch_ix = inxcoords.PatchCoords[x], patch_iw = inxcoords.PatchSizes[x], block_ix = inxcoords.BlockCoords[x];
                    int patch_ox = RemapCoord(patch_ix, inmap.Width, outmap.Width);
                    int patch_ow = RemapCoord(patch_iw, inmap.Width, outmap.Width);
                    int block_ox = RemapCoord(block_ix, inmap.Width, outmap.Width);

                    NdimArray<float> inblock = NdimArray<float>.Slice2D(inmap, block_ix, block_iw, block_iy, block_ih);
                    input.State = inblock;
                    flow.Execute();
                    NdimArray<float> outblock = output.State;

                    NdimArray<float>.RegionCopy2D(outblock, outmap, 
                        patch_ox - block_ox, patch_ox, patch_ow, 
                        patch_oy - block_oy, patch_oy, patch_oh);

                    ProgressEvent?.Invoke(i + 1, n);
                }
            }
        }

        /// <summary>計算実行(3D)</summary>
        private void Execute3D(NdimArray<float> inmap, NdimArray<float> outmap, PatchworkCoord inxcoords, PatchworkCoord inycoords, PatchworkCoord inzcoords) {
            int block_id = inzcoords.BlockSize, block_ih = inycoords.BlockSize, block_iw = inxcoords.BlockSize;

            int n = inzcoords.Blocks * inycoords.Blocks * inxcoords.Blocks;

            for (int i = 0, z = 0; z < inzcoords.Blocks; z++) {
                int patch_iz = inzcoords.PatchCoords[z], patch_id = inzcoords.PatchSizes[z], block_iz = inzcoords.BlockCoords[z];
                int patch_oz = RemapCoord(patch_iz, inmap.Height, outmap.Height);
                int patch_od = RemapCoord(patch_id, inmap.Height, outmap.Height);
                int block_oz = RemapCoord(block_iz, inmap.Height, outmap.Height);

                for (int y = 0; y < inycoords.Blocks; y++) {
                    int patch_iy = inycoords.PatchCoords[y], patch_ih = inycoords.PatchSizes[y], block_iy = inycoords.BlockCoords[y];
                    int patch_oy = RemapCoord(patch_iy, inmap.Height, outmap.Height);
                    int patch_oh = RemapCoord(patch_ih, inmap.Height, outmap.Height);
                    int block_oy = RemapCoord(block_iy, inmap.Height, outmap.Height);

                    for (int x = 0; x < inxcoords.Blocks; x++, i++) {
                        int patch_ix = inxcoords.PatchCoords[x], patch_iw = inxcoords.PatchSizes[x], block_ix = inxcoords.BlockCoords[x];
                        int patch_ox = RemapCoord(patch_ix, inmap.Width, outmap.Width);
                        int patch_ow = RemapCoord(patch_iw, inmap.Width, outmap.Width);
                        int block_ox = RemapCoord(block_ix, inmap.Width, outmap.Width);

                        NdimArray<float> inblock = NdimArray<float>.Slice3D(inmap, block_ix, block_iw, block_iy, block_ih, block_iz, block_id);
                        input.State = inblock;
                        flow.Execute();
                        NdimArray<float> outblock = output.State;

                        NdimArray<float>.RegionCopy3D(outblock, outmap,
                            patch_ox - block_ox, patch_ox, patch_ow,
                            patch_oy - block_oy, patch_oy, patch_oh,
                            patch_oz - block_oz, patch_oz, patch_od);

                        ProgressEvent?.Invoke(i + 1, n);
                    }
                }
            }
        }

        /// <summary>座標変換</summary>
        private static int RemapCoord(int x, int srcsize, int dstsize) {
            return (dstsize >= srcsize) ? (x * (dstsize / srcsize)) : (x / (srcsize / dstsize));
        }
    }
}
