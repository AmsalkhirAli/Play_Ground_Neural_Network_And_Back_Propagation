using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks.Dataflow;
namespace BackPropagation;

public class MLP
{
    public List<Layer> Layers { get; set; } = new List<Layer>();
    public Value Loss;

    public void CreateLayers(int neuronsCount, bool isInputLayer, bool isOutputLayer)
    {
        Layer Layer = new Layer(neuronsCount, isInputLayer, isOutputLayer);
        Layers.Add(Layer);
    }

    public void InitialiseNeuralNetwork(int features)
    {
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i].IsInputLayer)
            {
                foreach (var neuron in Layers[i].Neurons)
                {
                    neuron.InitialiseWeights(features);
                }
            }
            else
            {
                foreach (var neuron in Layers[i].Neurons)
                {
                    var isHiddenlayer = i != Layers.Count - 1;
                    neuron.InitialiseWeights(Layers[i - 1].NeuronsCount, isHiddenlayer);
                }
            }
        }
    }

    public Value ComputeLoss(List<double> targets)
    {

        var outputNeurons = Layers.Last().Neurons;
        Value total = null;

        for (int i = 0; i < outputNeurons.Count; i++)
        {
            var target = new Value(targets[i], $"target{i}");
            var perOutputLoss = outputNeurons[i].Output.LossCalc(target, $"L{i}");
            total = total == null ? perOutputLoss : total.Add(perOutputLoss, "");
        }

        Loss = total;
        return total;
    }

    public void TrainMlp(Dictionary<List<double>, List<double>> trainingData, int epochs, double learningRate, int batchsize)
    {
        bool didGradCheckThisEpoch = false;
        var parameters = GetParameters();

        // split the trainingdata in batches
        // first transform to a list
        List<(List<double>, List<double>)> trainingdataList = new List<(List<double>, List<double>)>();
        trainingdataList = trainingData.Select(kvp => (Inputs: kvp.Key, Outputs: kvp.Value)).ToList();



        for (int epoch = 0; epoch < epochs; epoch++)
        {
            trainingdataList.Shuffle();

            var batches = trainingdataList.Chunk(batchsize);

            double epochLoss = 0;

            foreach (var batch in batches)
            {
                double BatchLoss = 0.0;
                foreach (var sample in batch)
                {
                    var inputs = sample.Item1.Select((d, i) => new Value(d, $"in{i}")).ToList();
                    RunForward(inputs);
                    Loss = ComputeLoss(sample.Item2);
                    BatchLoss +=Loss.Data;
                    Loss.RunBackward();

                    if (epoch == 500 && !didGradCheckThisEpoch)
                    {
                        Helpers.NumericalGradCheck(Layers[0].Neurons[0].Weights[0], () => RunForward(inputs), sample, this, 1e-5);
                        didGradCheckThisEpoch = true;
                    }
                }

                double currentLr = learningRate * Math.Pow(0.9, epoch / 50);

                foreach (var p in parameters)
                    p.AdjustWeight(currentLr);

                Loss.ResetGradients();

                epochLoss += BatchLoss;

            }
            epochLoss = Loss.Data;
            if (epoch % 10 == 0 || epoch == epochs - 1)
                Console.WriteLine($"epoch {epoch,4}: avg loss = {epochLoss / trainingData.Count:F5}");

        }

        
    }



    List<Value> GetParameters()
    {
        var parameters = new List<Value>();
        foreach (var layer in Layers)
            foreach (var neuron in layer.Neurons)
            {
                parameters.AddRange(neuron.Weights);
                parameters.Add(neuron.Bias);
            }

        return parameters;
    }

    void RunForward(List<Value> values)
    {

        List<Value> dataFromLayer = null;
        for (int j = 0; j < Layers.Count; j++)
        {

            if (!Layers[j].IsInputLayer && dataFromLayer != null)
            {
                for (int i = 0; i < Layers[j].NeuronsCount; i++)
                {
                    var isHiddenlayer = j != 0 && j != Layers.Count - 1;
                    Layers[j].Neurons[i].Forward(dataFromLayer, isHiddenlayer);
                }
            }

            else
            {
                for (int i = 0; i < Layers[j].NeuronsCount; i++)
                {
                    Layers[j].Neurons[i].Forward(values);
                }

            }

            dataFromLayer = Layers[j].Neurons.Select(o => o.Output).ToList();

        }


    }

}