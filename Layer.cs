using System.Collections.Generic;

namespace BackPropagation;

public class Layer
{
    public bool IsInputLayer { get; set; } = false;
    public bool IsOutputLayer { get; set; } = false;
    public List<Neuron> Neurons { get; private set; } = new List<Neuron>();
    public int NeuronsCount => Neurons.Count;
    public List<Value> Inputs { get; set; }
    public List<Value> Outputs { get; set; }

    public Layer(int neuronsCount, bool isInputLayer = false, bool isOutputLayer = false)
    {
        for (int i = 0; i < neuronsCount; i++)
        {
            Neurons.Add(new Neuron());
        }
        IsInputLayer = isInputLayer;
        IsOutputLayer = isOutputLayer;
        
    }

    public void InitialiseWeightsInNeurons(int inputsCount)
    {
        foreach (var neuron in Neurons)
        {
            neuron.InitialiseWeights(inputsCount);
        }
    }
    
    
}