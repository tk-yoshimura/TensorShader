using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShaderTest {
    public class TestCaseIterator {
        private readonly List<int[]> cases;

        public int Times { private set; get; }
        public int Items { private set; get; }

        public TestCaseIterator(int times, params int[][] cases) {
            if (times < 1) {
                throw new ArgumentException(null, nameof(times));
            }
            if (cases is null || cases.Length == 0 || cases.Any((i) => i.Length < 1)) {
                throw new ArgumentException(null, nameof(cases));
            }

            this.cases = new List<int[]>();
            this.cases.AddRange(cases);

            this.Times = times;
            this.Items = cases.Length;
        }

        public IEnumerator<int[]> GetEnumerator() {
            Random random = new(123);

            int[] c = null;

            c = new int[Items];
            for (int i = 0; i < Items; i++) {
                c[i] = cases[i].First();
            }
            yield return c;

            List<int[]> counts = new();
            foreach (int[] item in cases) {
                counts.Add(new int[item.Length]);
            }

            bool is_finished = false;
            while (!is_finished) {
                is_finished = true;

                c = new int[Items];
                for (int i = 0; i < Items; i++) {
                    if (is_finished && counts[i].Any((count) => count < Times)) {
                        is_finished = false;
                    }

                    int min_count = counts[i].Min();

                    bool is_selected = false;
                    for (int j = 0; j < counts[i].Length; j++) {
                        if (counts[i][j] <= min_count && random.Next(4) < 1) {
                            c[i] = cases[i][j];
                            counts[i][j]++;
                            is_selected = true;
                            break;
                        }
                    }
                    if (!is_selected) {
                        int index = random.Next(cases[i].Length);
                        c[i] = cases[i][index];
                        counts[i][index]++;
                    }
                }

                yield return c;
            }

            c = new int[Items];
            for (int i = 0; i < Items; i++) {
                c[i] = cases[i].Last();
            }
            yield return c;

            yield break;
        }
    }
}
