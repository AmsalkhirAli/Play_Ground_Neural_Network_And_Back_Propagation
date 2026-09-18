using System;
using System.Diagnostics;
using System.IO;

namespace BackPropagation
{
    class Program
    {
        static void Main(string[] args)
        {


            Value x1 = new Value(2.0, "x1");
            Value w1 = new Value(-0.5, "w1");
            Value Bias = new Value(0.1, "bias");
            Value x2 = new Value(1.5, "x2");
            Value w2 = new Value(0.3, "w2");
            Value expected = new Value(0.5, "Expected");

            for (int epoch = 0; epoch < 20; epoch++)
            {
                // forward pass — rebuilds the graph fresh each time
                Value x1w1 = x1.Multiply(w1, "x1w1");
                Value x2w2 = x2.Multiply(w2, "x2w2");
                Value a = x1w1.Add(x2w2, "a");
                Value b = a.Add(Bias, "b");
                Value o = b.tanh("o");
                Value L = o.LossCalc(expected, "L");
                L.ResetGradients();

                // backward pass
                L.RunBackward();

                // update
                w1.AdjustWeight(0.01);
                w2.AdjustWeight(0.01);
                Bias.AdjustWeight(0.01);

                Console.WriteLine($"epoch {epoch}: loss = {L.Data:F4}");
            }



           
            Console.ReadKey();
        }
    }
}
