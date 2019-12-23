using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TensorShader;
using TensorShader.Initializers;
using TensorShader.Updaters.OptimizeMethod;
using TensorShader.Updaters.WeightDecay;
using TensorShaderCudaBackend.API;
using TensorShaderUtil.Iterator;
using TensorShaderUtil.SnapshotSaver;
using static TensorShader.Field;
using static TensorShader.VariableNode;

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
            VariableField x = new Tensor(loader.BatchShape);
            VariableField t = new Tensor(Shape.Vector(loader.NumBatches));

            Console.WriteLine("Build model...");
            Field y = CNN.Forward(x, classes);
            Field acc = Accuracy(y, t);
            Field err = Sum(SoftmaxCrossEntropy(y, OneHotVector(t, classes)), axes:new int[] { Axis.Map0D.Channels });
            StoreField accnode = acc.Save(), lossnode = Average(err).Save();

            Console.WriteLine("Build optimize flow...");
            (Flow trainflow, Parameters parameters) = Flow.Optimize(err);

            Console.WriteLine("Initialize params...");
            parameters
                .Where((parameter) => parameter.Category == ParameterCategory.Kernel)
                .InitializeTensor((tensor) => new HeNormal(tensor, random));
            parameters
                .Where((parameter) => parameter.Category == ParameterCategory.Bias)
                .InitializeTensor((tensor) => new Zero(tensor));

            Console.WriteLine("Set params updater...");
            parameters.AddUpdater((parameter) => new Nadam(parameter, alpha:0.01f));
            parameters.AddUpdater((parameter) => new Ridge(parameter, decay:1e-4f));

            Console.WriteLine("Training...");
            Train(train_iterator, loader, x, t, accnode, lossnode, trainflow, parameters);

            Console.WriteLine("Build inference flow...");
            Flow testflow = Flow.Inference(accnode);

            Console.WriteLine("Testing...");
            Test(test_iterator, loader, x, t, testflow, accnode);

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

        static void Train(Iterator train_iterator, MnistLoader loader, VariableField x, VariableField t, StoreField accnode, StoreField lossnode, Flow trainflow, Parameters parameters) {
            Stopwatch sw = new Stopwatch();

            sw.Start();

            while (train_iterator.Epoch < 50) {
                (float[] images, float[] labels) = loader.GetTrain(train_iterator.Next());

                x.ValueTensor.State = images;
                t.ValueTensor.State = labels;

                trainflow.Execute();
                parameters.Update();

                if (train_iterator.Iteration % 60 != 0) {
                    continue;
                }

                float train_acc = accnode.State[0];
                float train_loss = lossnode.State[0];

                Console.WriteLine($"[{train_iterator.Iteration}] train acc: {train_acc:F3} train loss: {train_loss:E3}");
            }

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec, {sw.ElapsedMilliseconds / train_iterator.Epoch} msec/epoch");
        }

        static void Test(Iterator test_iterator, MnistLoader loader, VariableField x, VariableField t, Flow testflow, StoreField accnode) {
            List<float> test_acc_list = new List<float>();

            Stopwatch sw = new Stopwatch();

            sw.Start();

            while (test_iterator.Epoch < 1) {
                (float[] images, float[] labels) = loader.GetTest(test_iterator.Next());

                x.ValueTensor.State = images;
                t.ValueTensor.State = labels;

                testflow.Execute();

                float test_acc = accnode.State[0];

                test_acc_list.Add(test_acc);
            }

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");

            Console.WriteLine($"[Result] acc: {test_acc_list.Average():F5}");
        }
    }
}
