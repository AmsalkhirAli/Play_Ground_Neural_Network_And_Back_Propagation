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
            double learningRate = 0.0025;
            int epochs = 800;

            mlp.TrainMlp(trainingData, epochs, learningRate,5);


            

            Console.ReadKey();
        }

        

      

       
    }
}
