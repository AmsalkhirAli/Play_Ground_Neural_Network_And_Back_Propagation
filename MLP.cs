using System;
using System.Collections.Generic;
using System.Linq;
namespace BackPropagation;

public class MLP
{
    public List<Layer> Layers { get; set; } = new List<Layer>();
    public Value Loss;

    public void CreateLayers(int neuronsCount,bool isInputLayer,bool isOutputLayer)
    {
        Layer Layer = new Layer(neuronsCount,isInputLayer, isOutputLayer);
        Layers.Add(Layer);
    }

    public void InitialiseNeuralNetwork()
    {
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i].IsInputLayer)
            {
                foreach (var neuron in Layers[i].Neurons)
                {
                    neuron.InitialiseWeights(1);
                }
            }
            else
            {
                foreach (var neuron in Layers[i].Neurons)
                {
                     neuron.InitialiseWeights(Layers[i - 1].NeuronsCount);
                }
            }
        }
    }

    public void RunForward(List<Value> values)
    {
        
        List<Value> dataFromLayer = null;
        for (int j = 0; j < Layers.Count; j++)
        {

            if (!Layers[j].IsInputLayer && dataFromLayer!=null)
            {
                for (int i = 0; i < Layers[j].NeuronsCount; i++)
                {
                    Layers[j].Neurons[i].Forward(dataFromLayer);
                }
            }
            
            else
            {
                for (int i = 0; i < Layers[j].NeuronsCount; i++)
                {
                    Layers[j].Neurons[i].Forward(new List<Value> { values[i] });
                }
                
            }
            
            dataFromLayer = Layers[j].Neurons.Select(o => o.Output).ToList();

        }
        
        
    }

    public void CalculateLoss(double expectedOutput)
    {
        
        Value ExpectedOutPut = new Value(expectedOutput,"L");
        
        Value AmountOfValuesForDivide = new Value(Layers.Where(o => o.IsOutputLayer)
            .First().NeuronsCount, "D");
        
        var OutputData = new List<Value>();
        
        Layers.Where(o => o.IsOutputLayer)
            .First()
            .Neurons
            .Select(n => n.Output)
            .ToList()
            .ForEach(o => OutputData.Add(o));



        Value FirstNeuronOutput = Layers.Where(o => o.IsOutputLayer)
            .First().Neurons[0].Output;

        Value total = null;
        foreach (var val in OutputData)
        {
            total = total == null ? val : total.Add(val, "");
        }

        total = total.Divide(AmountOfValuesForDivide,"");

        total = total.LossCalc(ExpectedOutPut, "L");
        Loss = total;
        
        
    }

    public void RunBackward()
    {
        Loss.RunBackward();
    }

}