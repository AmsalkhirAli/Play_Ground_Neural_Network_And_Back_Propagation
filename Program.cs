using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace BackPropagation
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlp = new MLP();
            mlp.Layers.Add(new Layer(6, true)); // input
            mlp.Layers.Add(new Layer(8, false)); // hidden 1
            mlp.Layers.Add(new Layer(8, false)); // hidden 2
            mlp.Layers.Add(new Layer(6, isOutputLayer: true)); // output

            mlp.InitialiseNeuralNetwork(); 
            
            var trainingData = new List<(List<double> Inputs, List<double> Targets)>
            {
                (new List<double> { 0.2, -0.1, 0.5, 0.3, -0.4, 0.1 },
                    new List<double> { 0.2, -0.1, 0.5, 0.3, -0.4, 0.1 }),
                (new List<double> { -0.3, 0.6, 0.1, -0.2, 0.4, -0.5 },
                    new List<double> { -0.3, 0.6, 0.1, -0.2, 0.4, -0.5 }),
                (new List<double> { 0.5, 0.5, -0.5, -0.5, 0.2, 0.2 },
                    new List<double> { 0.5, 0.5, -0.5, -0.5, 0.2, 0.2 }),
                (new List<double> { -0.1, -0.6, 0.4, 0.4, -0.3, 0.6 },
                    new List<double> { -0.1, -0.6, 0.4, 0.4, -0.3, 0.6 }),
            };

            var parameters = GetParameters(mlp); 
            double learningRate = 0.09;
            int epochs = 800;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double epochLoss = 0;

                foreach (var sample in trainingData)
                {
                    var inputs = sample.Inputs.Select((d, i) => new Value(d, $"in{i}")).ToList();
                    mlp.RunForward(inputs);

                    var loss = ComputeLoss(mlp, sample.Targets);

                    loss.ResetGradients();
                    loss.RunBackward();

                    foreach (var p in parameters)
                        p.AdjustWeight(learningRate);

                    epochLoss += loss.Data;
                }

                if (epoch % 10 == 0 || epoch == epochs - 1)
                    Console.WriteLine($"epoch {epoch,4}: avg loss = {epochLoss / trainingData.Count:F5}");
            }

            Console.ReadKey();
        }

        static List<Value> GetParameters(MLP mlp)
        {
            var parameters = new List<Value>();
            foreach (var layer in mlp.Layers)
            foreach (var neuron in layer.Neurons)
            {
                parameters.AddRange(neuron.Weights);
                parameters.Add(neuron.Bias);
            }

            return parameters;
        }

        static Value ComputeLoss(MLP mlp, List<double> targets)
        {

            var outputNeurons = mlp.Layers.Last().Neurons;
            Value total = null;

            for (int i = 0; i < outputNeurons.Count; i++)
            {
                var target = new Value(targets[i], $"target{i}");
                var perOutputLoss = outputNeurons[i].Output.LossCalc(target, $"L{i}");
                total = total == null ? perOutputLoss : total.Add(perOutputLoss, "");
            }

            mlp.Loss = total;
            return total;
        }
    }
}
