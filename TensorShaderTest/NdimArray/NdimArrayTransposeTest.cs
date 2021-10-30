using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using TensorShader;

namespace TensorShaderTest.NdimArray {
    [TestClass]
    public class NdimArrayTransposeTest {
        [TestMethod]
        public void Dim2Test() {
            NdimArray<int> arr = new NdimArray<int>(
                new Shape(ShapeType.Undefined, 2, 3),
                (new int[6]).Select((_, i) => i).ToArray()
            );

            NdimArray<int> arr1 = NdimArray<int>.Transpose(arr, (0, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 2, 3), arr1.Shape);
            CollectionAssert.AreEqual(new int[] {
                0, 1, 2, 3, 4, 5
            }, arr1.Value);

            NdimArray<int> arr2 = NdimArray<int>.Transpose(arr, (1, 0));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 3, 2), arr2.Shape);
            CollectionAssert.AreEqual(new int[] {
                0, 2, 4, 1, 3, 5
            }, arr2.Value);
        }

        [TestMethod]
        public void Dim3Test() {
            NdimArray<int> arr = new NdimArray<int>(
                new Shape(ShapeType.Undefined, 2, 3, 4),
                (new int[24]).Select((_, i) => i).ToArray()
            );

            NdimArray<int> arr1 = NdimArray<int>.Transpose(arr, (0, 1, 2));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 2, 3, 4), arr1.Shape);
            CollectionAssert.AreEqual(new int[] {
                0, 1, 2, 3, 4, 5,
                6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23,
            }, arr1.Value);
            
            NdimArray<int> arr2 = NdimArray<int>.Transpose(arr, (1, 0, 2));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 3, 2, 4), arr2.Shape);
            CollectionAssert.AreEqual(new int[] {
                0 , 2 , 4 , 1 , 3 , 5 ,
                6 , 8 , 10, 7 , 9 , 11,
                12, 14, 16, 13, 15, 17,
                18, 20, 22, 19, 21, 23,
            }, arr2.Value);
            
            NdimArray<int> arr3 = NdimArray<int>.Transpose(arr, (2, 1, 0));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 4, 3, 2), arr3.Shape);
            CollectionAssert.AreEqual(new int[] {
                 0  ,6 ,12 ,18 , 2 ,8  ,
                 14 ,20,  4, 10, 16,22 ,
                 1  ,7 ,13 ,19 , 3 ,9  ,
                 15 ,21,  5, 11, 17,23 ,
            }, arr3.Value);

            NdimArray<int> arr4 = NdimArray<int>.Transpose(arr, (1, 2, 0));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 3, 4, 2), arr4.Shape);
            CollectionAssert.AreEqual(new int[] {
                 0 ,  2,  4,  6,  8, 10,
                 12, 14, 16, 18, 20, 22,
                  1,  3,  5,  7,  9, 11,
                 13, 15, 17, 19, 21, 23,
            }, arr4.Value);

            NdimArray<int> arr5 = NdimArray<int>.Transpose(arr, (0, 2, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 2, 4, 3), arr5.Shape);
            CollectionAssert.AreEqual(new int[] {
                  0 , 1 ,  6,  7, 12, 13,
                  18, 19,  2,  3,  8,  9,
                  14, 15, 20, 21,  4,  5,
                  10, 11, 16, 17, 22, 23,
            }, arr5.Value);

            NdimArray<int> arr6 = NdimArray<int>.Transpose(arr, (2, 0, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 4, 2, 3), arr6.Shape);
            CollectionAssert.AreEqual(new int[] {
                   0 , 6 , 12, 18,  1,  7,
                   13, 19,  2,  8, 14, 20,
                   3 , 9 , 15, 21,  4, 10,
                   16, 22,  5, 11, 17, 23,
            }, arr6.Value);
        }

        [TestMethod]
        public void Dim4Test() {
            NdimArray<int> arr = new NdimArray<int>(
                new Shape(ShapeType.Undefined, 2, 3, 4, 5),
                (new int[120]).Select((_, i) => i).ToArray()
            );

            NdimArray<int> arr1 = NdimArray<int>.Transpose(arr, (0, 1, 2, 3));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 2, 3, 4, 5), arr1.Shape);
            CollectionAssert.AreEqual(new int[] {
                 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,100,101,102,103,104,105,106,107,
                108,109,110,111,112,113,114,115,116,117,118,119
            }, arr1.Value);

            NdimArray<int> arr2 = NdimArray<int>.Transpose(arr, (2, 3, 0, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 4, 5, 2, 3), arr2.Shape);
            CollectionAssert.AreEqual(new int[] {
                   0,  6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96,102,
                 108,114,  1,  7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91,
                  97,103,109,115,  2,  8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80,
                  86, 92, 98,104,110,116,  3,  9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69,
                  75, 81, 87, 93, 99,105,111,117,  4, 10, 16, 22, 28, 34, 40, 46, 52, 58,
                  64, 70, 76, 82, 88, 94,100,106,112,118,  5, 11, 17, 23, 29, 35, 41, 47,
                  53, 59, 65, 71, 77, 83, 89, 95,101,107,113,119
            }, arr2.Value);

            NdimArray<int> arr3 = NdimArray<int>.Transpose(arr, (2, 0, 3, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 4, 2, 5, 3), arr3.Shape);
            CollectionAssert.AreEqual(new int[] {
                   0,  6, 12, 18,  1,  7, 13, 19, 24, 30, 36, 42, 25, 31, 37, 43, 48, 54,
                  60, 66, 49, 55, 61, 67, 72, 78, 84, 90, 73, 79, 85, 91, 96,102,108,114,
                  97,103,109,115,  2,  8, 14, 20,  3,  9, 15, 21, 26, 32, 38, 44, 27, 33,
                  39, 45, 50, 56, 62, 68, 51, 57, 63, 69, 74, 80, 86, 92, 75, 81, 87, 93,
                  98,104,110,116, 99,105,111,117,  4, 10, 16, 22,  5, 11, 17, 23, 28, 34,
                  40, 46, 29, 35, 41, 47, 52, 58, 64, 70, 53, 59, 65, 71, 76, 82, 88, 94,
                  77, 83, 89, 95,100,106,112,118,101,107,113,119
            }, arr3.Value);

            NdimArray<int> arr4 = NdimArray<int>.Transpose(arr, (1, 0, 3, 2));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 3, 2, 5, 4), arr4.Shape);
            CollectionAssert.AreEqual(new int[] {
                   0,  2,  4,  1,  3,  5, 24, 26, 28, 25, 27, 29, 48, 50, 52, 49, 51, 53,
                  72, 74, 76, 73, 75, 77, 96, 98,100, 97, 99,101,  6,  8, 10,  7,  9, 11,
                  30, 32, 34, 31, 33, 35, 54, 56, 58, 55, 57, 59, 78, 80, 82, 79, 81, 83,
                 102,104,106,103,105,107, 12, 14, 16, 13, 15, 17, 36, 38, 40, 37, 39, 41,
                  60, 62, 64, 61, 63, 65, 84, 86, 88, 85, 87, 89,108,110,112,109,111,113,
                  18, 20, 22, 19, 21, 23, 42, 44, 46, 43, 45, 47, 66, 68, 70, 67, 69, 71,
                  90, 92, 94, 91, 93, 95,114,116,118,115,117,119
            }, arr4.Value);

            NdimArray<int> arr5 = NdimArray<int>.Transpose(arr, (3, 2, 1, 0));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 5, 4, 3, 2), arr5.Shape);
            CollectionAssert.AreEqual(new int[] {
                   0, 24, 48, 72, 96,  6, 30, 54, 78,102, 12, 36, 60, 84,108, 18, 42, 66,
                  90,114,  2, 26, 50, 74, 98,  8, 32, 56, 80,104, 14, 38, 62, 86,110, 20,
                  44, 68, 92,116,  4, 28, 52, 76,100, 10, 34, 58, 82,106, 16, 40, 64, 88,
                 112, 22, 46, 70, 94,118,  1, 25, 49, 73, 97,  7, 31, 55, 79,103, 13, 37,
                  61, 85,109, 19, 43, 67, 91,115,  3, 27, 51, 75, 99,  9, 33, 57, 81,105,
                  15, 39, 63, 87,111, 21, 45, 69, 93,117,  5, 29, 53, 77,101, 11, 35, 59,
                  83,107, 17, 41, 65, 89,113, 23, 47, 71, 95,119
            }, arr5.Value);

            NdimArray<int> arr6 = NdimArray<int>.Transpose(arr, (3, 1, 2, 0));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 5, 3, 4, 2), arr6.Shape);
            CollectionAssert.AreEqual(new int[] {
                  0, 24, 48, 72, 96,  2, 26, 50, 74, 98,  4, 28, 52, 76,100,  6, 30, 54,
                  78,102,  8, 32, 56, 80,104, 10, 34, 58, 82,106, 12, 36, 60, 84,108, 14,
                  38, 62, 86,110, 16, 40, 64, 88,112, 18, 42, 66, 90,114, 20, 44, 68, 92,
                 116, 22, 46, 70, 94,118,  1, 25, 49, 73, 97,  3, 27, 51, 75, 99,  5, 29,
                  53, 77,101,  7, 31, 55, 79,103,  9, 33, 57, 81,105, 11, 35, 59, 83,107,
                  13, 37, 61, 85,109, 15, 39, 63, 87,111, 17, 41, 65, 89,113, 19, 43, 67,
                  91,115, 21, 45, 69, 93,117, 23, 47, 71, 95,119
            }, arr6.Value);
        }

        [TestMethod]
        public void Dim5Test() {
            NdimArray<int> arr = new NdimArray<int>(
                new Shape(ShapeType.Undefined, 2, 3, 4, 5, 6),
                (new int[720]).Select((_, i) => i).ToArray()
            );

            NdimArray<int> arr1 = NdimArray<int>.Transpose(arr, (2, 0, 3, 4, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 4, 2, 5, 6, 3), arr1.Shape);
            CollectionAssert.AreEqual(new int[] {
                   0,  6, 12, 18,  1,  7, 13, 19, 24, 30, 36, 42, 25, 31, 37, 43, 48, 54,
                  60, 66, 49, 55, 61, 67, 72, 78, 84, 90, 73, 79, 85, 91, 96,102,108,114,
                  97,103,109,115,120,126,132,138,121,127,133,139,144,150,156,162,145,151,
                 157,163,168,174,180,186,169,175,181,187,192,198,204,210,193,199,205,211,
                 216,222,228,234,217,223,229,235,240,246,252,258,241,247,253,259,264,270,
                 276,282,265,271,277,283,288,294,300,306,289,295,301,307,312,318,324,330,
                 313,319,325,331,336,342,348,354,337,343,349,355,360,366,372,378,361,367,
                 373,379,384,390,396,402,385,391,397,403,408,414,420,426,409,415,421,427,
                 432,438,444,450,433,439,445,451,456,462,468,474,457,463,469,475,480,486,
                 492,498,481,487,493,499,504,510,516,522,505,511,517,523,528,534,540,546,
                 529,535,541,547,552,558,564,570,553,559,565,571,576,582,588,594,577,583,
                 589,595,600,606,612,618,601,607,613,619,624,630,636,642,625,631,637,643,
                 648,654,660,666,649,655,661,667,672,678,684,690,673,679,685,691,696,702,
                 708,714,697,703,709,715,  2,  8, 14, 20,  3,  9, 15, 21, 26, 32, 38, 44,
                  27, 33, 39, 45, 50, 56, 62, 68, 51, 57, 63, 69, 74, 80, 86, 92, 75, 81,
                  87, 93, 98,104,110,116, 99,105,111,117,122,128,134,140,123,129,135,141,
                 146,152,158,164,147,153,159,165,170,176,182,188,171,177,183,189,194,200,
                 206,212,195,201,207,213,218,224,230,236,219,225,231,237,242,248,254,260,
                 243,249,255,261,266,272,278,284,267,273,279,285,290,296,302,308,291,297,
                 303,309,314,320,326,332,315,321,327,333,338,344,350,356,339,345,351,357,
                 362,368,374,380,363,369,375,381,386,392,398,404,387,393,399,405,410,416,
                 422,428,411,417,423,429,434,440,446,452,435,441,447,453,458,464,470,476,
                 459,465,471,477,482,488,494,500,483,489,495,501,506,512,518,524,507,513,
                 519,525,530,536,542,548,531,537,543,549,554,560,566,572,555,561,567,573,
                 578,584,590,596,579,585,591,597,602,608,614,620,603,609,615,621,626,632,
                 638,644,627,633,639,645,650,656,662,668,651,657,663,669,674,680,686,692,
                 675,681,687,693,698,704,710,716,699,705,711,717,  4, 10, 16, 22,  5, 11,
                  17, 23, 28, 34, 40, 46, 29, 35, 41, 47, 52, 58, 64, 70, 53, 59, 65, 71,
                  76, 82, 88, 94, 77, 83, 89, 95,100,106,112,118,101,107,113,119,124,130,
                 136,142,125,131,137,143,148,154,160,166,149,155,161,167,172,178,184,190,
                 173,179,185,191,196,202,208,214,197,203,209,215,220,226,232,238,221,227,
                 233,239,244,250,256,262,245,251,257,263,268,274,280,286,269,275,281,287,
                 292,298,304,310,293,299,305,311,316,322,328,334,317,323,329,335,340,346,
                 352,358,341,347,353,359,364,370,376,382,365,371,377,383,388,394,400,406,
                 389,395,401,407,412,418,424,430,413,419,425,431,436,442,448,454,437,443,
                 449,455,460,466,472,478,461,467,473,479,484,490,496,502,485,491,497,503,
                 508,514,520,526,509,515,521,527,532,538,544,550,533,539,545,551,556,562,
                 568,574,557,563,569,575,580,586,592,598,581,587,593,599,604,610,616,622,
                 605,611,617,623,628,634,640,646,629,635,641,647,652,658,664,670,653,659,
                 665,671,676,682,688,694,677,683,689,695,700,706,712,718,701,707,713,719
            }, arr1.Value);

            NdimArray<int> arr2 = NdimArray<int>.Transpose(arr, (4, 0, 3, 1, 2));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 6, 2, 5, 3, 4), arr2.Shape);
            CollectionAssert.AreEqual(new int[] {
                   0,120,240,360,480,600,  1,121,241,361,481,601, 24,144,264,384,504,624,
                  25,145,265,385,505,625, 48,168,288,408,528,648, 49,169,289,409,529,649,
                  72,192,312,432,552,672, 73,193,313,433,553,673, 96,216,336,456,576,696,
                  97,217,337,457,577,697,  2,122,242,362,482,602,  3,123,243,363,483,603,
                  26,146,266,386,506,626, 27,147,267,387,507,627, 50,170,290,410,530,650,
                  51,171,291,411,531,651, 74,194,314,434,554,674, 75,195,315,435,555,675,
                  98,218,338,458,578,698, 99,219,339,459,579,699,  4,124,244,364,484,604,
                   5,125,245,365,485,605, 28,148,268,388,508,628, 29,149,269,389,509,629,
                  52,172,292,412,532,652, 53,173,293,413,533,653, 76,196,316,436,556,676,
                  77,197,317,437,557,677,100,220,340,460,580,700,101,221,341,461,581,701,
                   6,126,246,366,486,606,  7,127,247,367,487,607, 30,150,270,390,510,630,
                  31,151,271,391,511,631, 54,174,294,414,534,654, 55,175,295,415,535,655,
                  78,198,318,438,558,678, 79,199,319,439,559,679,102,222,342,462,582,702,
                 103,223,343,463,583,703,  8,128,248,368,488,608,  9,129,249,369,489,609,
                  32,152,272,392,512,632, 33,153,273,393,513,633, 56,176,296,416,536,656,
                  57,177,297,417,537,657, 80,200,320,440,560,680, 81,201,321,441,561,681,
                 104,224,344,464,584,704,105,225,345,465,585,705, 10,130,250,370,490,610,
                  11,131,251,371,491,611, 34,154,274,394,514,634, 35,155,275,395,515,635,
                  58,178,298,418,538,658, 59,179,299,419,539,659, 82,202,322,442,562,682,
                  83,203,323,443,563,683,106,226,346,466,586,706,107,227,347,467,587,707,
                  12,132,252,372,492,612, 13,133,253,373,493,613, 36,156,276,396,516,636,
                  37,157,277,397,517,637, 60,180,300,420,540,660, 61,181,301,421,541,661,
                  84,204,324,444,564,684, 85,205,325,445,565,685,108,228,348,468,588,708,
                 109,229,349,469,589,709, 14,134,254,374,494,614, 15,135,255,375,495,615,
                  38,158,278,398,518,638, 39,159,279,399,519,639, 62,182,302,422,542,662,
                  63,183,303,423,543,663, 86,206,326,446,566,686, 87,207,327,447,567,687,
                 110,230,350,470,590,710,111,231,351,471,591,711, 16,136,256,376,496,616,
                  17,137,257,377,497,617, 40,160,280,400,520,640, 41,161,281,401,521,641,
                  64,184,304,424,544,664, 65,185,305,425,545,665, 88,208,328,448,568,688,
                  89,209,329,449,569,689,112,232,352,472,592,712,113,233,353,473,593,713,
                  18,138,258,378,498,618, 19,139,259,379,499,619, 42,162,282,402,522,642,
                  43,163,283,403,523,643, 66,186,306,426,546,666, 67,187,307,427,547,667,
                  90,210,330,450,570,690, 91,211,331,451,571,691,114,234,354,474,594,714,
                 115,235,355,475,595,715, 20,140,260,380,500,620, 21,141,261,381,501,621,
                  44,164,284,404,524,644, 45,165,285,405,525,645, 68,188,308,428,548,668,
                  69,189,309,429,549,669, 92,212,332,452,572,692, 93,213,333,453,573,693,
                 116,236,356,476,596,716,117,237,357,477,597,717, 22,142,262,382,502,622,
                  23,143,263,383,503,623, 46,166,286,406,526,646, 47,167,287,407,527,647,
                  70,190,310,430,550,670, 71,191,311,431,551,671, 94,214,334,454,574,694,
                  95,215,335,455,575,695,118,238,358,478,598,718,119,239,359,479,599,719
            }, arr2.Value);
        }

        [TestMethod]
        public void Dim6Test() {
            NdimArray<int> arr = new NdimArray<int>(
                new Shape(ShapeType.Undefined, 2, 3, 4, 5, 6, 7),
                (new int[5040]).Select((_, i) => i).ToArray()
            );

            NdimArray<int> arr_transpose = NdimArray<int>.Transpose(arr, (2, 0, 3, 5, 4, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 4, 2, 5, 7, 6, 3), arr_transpose.Shape);

            for (int i5 = 0; i5 < arr.Shape[5]; i5++) { 
                for (int i4 = 0; i4 < arr.Shape[4]; i4++) {
                    for (int i3 = 0; i3 < arr.Shape[3]; i3++) {
                        for (int i2 = 0; i2 < arr.Shape[2]; i2++) {
                            for (int i1 = 0; i1 < arr.Shape[1]; i1++) {
                                for (int i0 = 0; i0 < arr.Shape[0]; i0++) {
                                    Assert.AreEqual(
                                        arr[i0, i1, i2, i3, i4, i5], 
                                        arr_transpose[i2, i0, i3, i5, i4, i1]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Dim7Test() {
            NdimArray<int> arr = new NdimArray<int>(
                new Shape(ShapeType.Undefined, 2, 3, 4, 5, 6, 7, 8),
                (new int[40320]).Select((_, i) => i).ToArray()
            );

            NdimArray<int> arr_transpose = NdimArray<int>.Transpose(arr, (2, 0, 6, 3, 5, 4, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 4, 2, 8, 5, 7, 6, 3), arr_transpose.Shape);

            for (int i6 = 0; i6 < arr.Shape[6]; i6++) {
                for (int i5 = 0; i5 < arr.Shape[5]; i5++) {
                    for (int i4 = 0; i4 < arr.Shape[4]; i4++) {
                        for (int i3 = 0; i3 < arr.Shape[3]; i3++) {
                            for (int i2 = 0; i2 < arr.Shape[2]; i2++) {
                                for (int i1 = 0; i1 < arr.Shape[1]; i1++) {
                                    for (int i0 = 0; i0 < arr.Shape[0]; i0++) {
                                        Assert.AreEqual(
                                            arr[i0, i1, i2, i3, i4, i5, i6], 
                                            arr_transpose[i2, i0, i6, i3, i5, i4, i1]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Dim8Test() {
            NdimArray<int> arr = new NdimArray<int>(
                new Shape(ShapeType.Undefined, 2, 3, 4, 5, 6, 7, 8, 9),
                (new int[362880]).Select((_, i) => i).ToArray()
            );

            NdimArray<int> arr_transpose = NdimArray<int>.Transpose(arr, (2, 0, 6, 3, 7, 5, 4, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 4, 2, 8, 5, 9, 7, 6, 3), arr_transpose.Shape);

            for (int i7 = 0; i7 < arr.Shape[7]; i7++) {
                for (int i6 = 0; i6 < arr.Shape[6]; i6++) {
                    for (int i5 = 0; i5 < arr.Shape[5]; i5++) {
                        for (int i4 = 0; i4 < arr.Shape[4]; i4++) {
                            for (int i3 = 0; i3 < arr.Shape[3]; i3++) {
                                for (int i2 = 0; i2 < arr.Shape[2]; i2++) {
                                    for (int i1 = 0; i1 < arr.Shape[1]; i1++) {
                                        for (int i0 = 0; i0 < arr.Shape[0]; i0++) {
                                            Assert.AreEqual(
                                                arr[i0, i1, i2, i3, i4, i5, i6, i7], 
                                                arr_transpose[i2, i0, i6, i3, i7, i5, i4, i1]
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void Axis1Test() {
            NdimArray<int> arr = new NdimArray<int>(
                new Shape(ShapeType.Undefined, 2, 3, 1, 5),
                (new int[30]).Select((_, i) => i).ToArray()
            );

            NdimArray<int> arr1 = NdimArray<int>.Transpose(arr, (0, 1, 2, 3));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 2, 3, 1, 5), arr1.Shape);
            CollectionAssert.AreEqual(new int[] {
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                 24,25,26,27,28,29
            }, arr1.Value);

            NdimArray<int> arr2 = NdimArray<int>.Transpose(arr, (2, 3, 0, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 1, 5, 2, 3), arr2.Shape);
            CollectionAssert.AreEqual(new int[] {
                  0, 6,12,18,24, 1, 7,13,19,25, 2, 8,14,20,26, 3, 9,15,21,27, 4,10,16,22,
                 28, 5,11,17,23,29
            }, arr2.Value);

            NdimArray<int> arr3 = NdimArray<int>.Transpose(arr, (2, 0, 3, 1));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 1, 2, 5, 3), arr3.Shape);
            CollectionAssert.AreEqual(new int[] {
                  0, 1, 6, 7,12,13,18,19,24,25, 2, 3, 8, 9,14,15,20,21,26,27, 4, 5,10,11,
                 16,17,22,23,28,29
            }, arr3.Value);

            NdimArray<int> arr4 = NdimArray<int>.Transpose(arr, (1, 0, 3, 2));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 3, 2, 5, 1), arr4.Shape);
            CollectionAssert.AreEqual(new int[] {
                   0, 2, 4, 1, 3, 5, 6, 8,10, 7, 9,11,12,14,16,13,15,17,18,20,22,19,21,23,
                  24,26,28,25,27,29
            }, arr4.Value);

            NdimArray<int> arr5 = NdimArray<int>.Transpose(arr, (3, 2, 1, 0));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 5, 1, 3, 2), arr5.Shape);
            CollectionAssert.AreEqual(new int[] {
                  0, 6,12,18,24, 2, 8,14,20,26, 4,10,16,22,28, 1, 7,13,19,25, 3, 9,15,21,
                 27, 5,11,17,23,29
            }, arr5.Value);

            NdimArray<int> arr6 = NdimArray<int>.Transpose(arr, (3, 1, 2, 0));
            Assert.AreEqual(new Shape(ShapeType.Undefined, 5, 3, 1, 2), arr6.Shape);
            CollectionAssert.AreEqual(new int[] {
                   0, 6,12,18,24, 2, 8,14,20,26, 4,10,16,22,28, 1, 7,13,19,25, 3, 9,15,21,
                  27, 5,11,17,23,29
            }, arr6.Value);
        }

        [TestMethod]
        public void ExceptionTest() {
            NdimArray<int> arr = new NdimArray<int>(
                new Shape(ShapeType.Undefined, 2, 3, 4),
                (new int[24]).Select((_, i) => i).ToArray()
            );

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int>.Transpose(arr, (1, 0));
            });

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int>.Transpose(arr, (3, 2, 1, 0));
            });

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int>.Transpose(arr, (1, 1, 0));
            });

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int>.Transpose(arr, (0, 0, 0));
            });

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int>.Transpose(arr, (1, -1, 0));
            });

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int>.Transpose(arr, (3, 2, 0));
            });

            Assert.ThrowsException<ArgumentException>(() => {
                NdimArray<int>.Transpose(arr, (3, 2, 1));
            });
        }
    }
}
