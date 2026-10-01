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
                    neuron.InitialiseWeights(Layers[i - 1].NeuronsCount,isHiddenlayer);
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
                    var isHiddenlayer = j!=0 && j!=Layers.Count-1;
                    Layers[j].Neurons[i].Forward(dataFromLayer,isHiddenlayer);
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