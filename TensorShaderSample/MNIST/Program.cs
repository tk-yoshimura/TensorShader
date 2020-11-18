using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

using TensorShader;
using TensorShader.Initializers;
using TensorShader.Updaters.OptimizeMethod;
using TensorShader.Updaters.WeightDecay;
using TensorShaderUtil.GraphVisualization;
using TensorShaderUtil.Iterator;
using TensorShaderUtil.SnapshotSaver;
using static TensorShader.Field;

namespace MNIST {
    class Program {
        static void Main() {
            const string dirpath_dataset = "mnist_dataset";
            const string dirpath_result = "result";
            const int classes = 10;

            Console.WriteLine("Download mnist...");
            MnistDownloader.Download(dirpath_dataset);

            Console.WriteLine("Setup loader...");
            Random random = new Random(1234);

            MnistLoader loader = new MnistLoader(dirpath_dataset, num_batches: 1000);
            Iterator train_iterator = new ShuffleIterator(loader.NumBatches, loader.CountTrainDatas, random);
            Iterator test_iterator = new ShuffleIterator(loader.NumBatches, loader.CountTestDatas, random);

            Console.WriteLine("Create input tensor...");
            VariableField x = loader.BatchShape;
            VariableField t = Shape.Vector(loader.NumBatches);

            Console.WriteLine("Build model...");
            Field y = Model.CNN(x, classes);
            StoreField acc = Accuracy(y, t);
            Field loss = Sum(
                    SoftmaxCrossEntropy(y, OneHotVector(t, classes)),
                    Axis.Map0D.Channels
                );
            StoreField avg_loss = Average(loss);

            Console.WriteLine("Visualize computation graph...");
            Graph.WriteDotFile("graph.dot", loss, acc);

            Console.WriteLine("Set iterator event...");
            train_iterator.IncreasedEpoch += (iter) => {
                float train_acc = (float)acc.State;
                float train_loss = (float)avg_loss.State;

                Console.WriteLine($"[{iter.Iteration}] train acc: {train_acc:F3} train loss: {train_loss:E3}");
            };

            Console.WriteLine("Build optimize flow...");
            (Flow trainflow, Parameters parameters) = Flow.Optimize(loss);

            Console.WriteLine("Initialize params...");
            parameters
                .Where((parameter) => parameter.Category == ParameterCategory.Kernel)
                .InitializeTensor((tensor) => new HeNormal(tensor, random));
            parameters
                .Where((parameter) => parameter.Category == ParameterCategory.Bias)
                .InitializeTensor((tensor) => new Zero(tensor));

            Console.WriteLine("Set params updater...");
            parameters.AddUpdater((parameter) => new Nadam(parameter, alpha: 0.01f));
            parameters.AddUpdater((parameter) => new Ridge(parameter, decay: 1e-4f));

            Console.WriteLine("Training...");
            Train(train_iterator, loader, x, t, trainflow, parameters);

            Console.WriteLine("Build inference flow...");
            (Flow testflow, _) = Flow.Inference(acc);

            Console.WriteLine("Testing...");
            Test(test_iterator, loader, x, t, testflow, acc);

            Console.WriteLine("Saving snapshot...");
            Snapshot snapshot = parameters.Save();
            SnapshotSaver saver = new ZippedBinaryShapshotSaver();

            if (!Directory.Exists(dirpath_result)) {
                Directory.CreateDirectory(dirpath_result);
            }
            saver.Save($"{dirpath_result}/mnist.tss", snapshot);

            Console.WriteLine("END");
            Console.Read();
        }

        static void Train(Iterator train_iterator, MnistLoader loader, VariableField x, VariableField t, Flow trainflow, Parameters parameters) {
            Stopwatch sw = new Stopwatch();

            sw.Start();

            while (train_iterator.Epoch < 5) {
                (NdimArray<float> images, NdimArray<float> labels) = loader.GetTrain(train_iterator.Next());

                x.State = images;
                t.State = labels;

                trainflow.Execute();
                parameters.Update();
            }

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec, {sw.ElapsedMilliseconds / train_iterator.Epoch} msec/epoch");
        }

        static void Test(Iterator test_iterator, MnistLoader loader, VariableField x, VariableField t, Flow testflow, StoreField acc) {
            List<float> test_acc_list = new List<float>();

            Stopwatch sw = new Stopwatch();

            sw.Start();

            while (test_iterator.Epoch < 1) {
                (NdimArray<float> images, NdimArray<float> labels) = loader.GetTest(test_iterator.Next());

                x.State = images;
                t.State = labels;

                testflow.Execute();

                float test_acc = (float)acc.State;

                test_acc_list.Add(test_acc);
            }

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");

            Console.WriteLine($"[Result] acc: {test_acc_list.Average():F5}");
        }
    }
}
