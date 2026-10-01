using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.Marshalling;


namespace BackPropagation
{
    class Program
    {
        
        static void Main(string[] args)
        {
            int inputFeatures = 6;
            var mlp = new MLP();
            mlp.Layers.Add(new Layer(6, true)); // input
            mlp.Layers.Add(new Layer(8, false)); // hidden 1
            mlp.Layers.Add(new Layer(8, false)); // hidden 2
            mlp.Layers.Add(new Layer(6, isOutputLayer: true)); // output

            mlp.InitialiseNeuralNetwork(inputFeatures);

            var trainingData = Helpers.getDataForTesting(inputFeatures, 50);


            var parameters = GetParameters(mlp); 
            double learningRate = 0.0015;
            int epochs = 800;
            bool didGradCheckThisEpoch = false;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double epochLoss = 0;

                foreach (var sample in trainingData)
                {
                    var inputs = sample.Key.Select((d, i) => new Value(d, $"in{i}")).ToList();
                    mlp.RunForward(inputs);

                    var loss = ComputeLoss(mlp, sample.Value);

                    loss.ResetGradients();
                    loss.RunBackward();

                    if (epoch == 500 && !didGradCheckThisEpoch)
                    {
                        NumericalGradCheck(mlp.Layers[0].Neurons[0].Weights[0], () => mlp.RunForward(inputs), sample, mlp, 1e-5);
                        didGradCheckThisEpoch = true;
                    }


                    foreach (var p in parameters)
                        p.AdjustWeight(learningRate);

                    epochLoss += loss.Data;
                }

                if (epoch % 10 == 0 || epoch == epochs - 1)
                    Console.WriteLine($"epoch {epoch,4}: avg loss = {epochLoss / trainingData.Count:F5}");
            }

            Console.ReadKey();
        }

        static void NumericalGradCheck(Value weight, Action rebuildForwardPass,KeyValuePair<List<double>,List<double>>sample,MLP mlp,double h = 1e-5)
        {
            // original gradient after 1 backwards run
            Console.WriteLine($"original gradient:{weight.Grad}");
            var OriginalWeight = weight.Data;
            weight.Data += h;
            rebuildForwardPass();
            var lossPlus = ComputeLoss(mlp, sample.Value);
            weight.Data -= 2*h;
            rebuildForwardPass();
            var lossMinus = ComputeLoss(mlp, sample.Value);
            var numericalGrad = (lossPlus.Data - lossMinus.Data) / (2 * h);
            weight.Data = OriginalWeight;

            Console.WriteLine($"NumericalGrad = {numericalGrad}  ::  BackPropGrad = {weight.Grad}");
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
