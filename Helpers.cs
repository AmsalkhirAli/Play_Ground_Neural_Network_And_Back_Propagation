using System;
using System.Collections.Generic;

namespace BackPropagation
{
    public class Helpers
    {
        public static Random randomGenerator = new Random(DateTime.Now.Millisecond);
        public static Dictionary<List<double>, List<double>> getDataForTesting(int inputsAvaliable, int amountOfDataBatches)
        {
            Dictionary<List<double>, List<double>> syntheticData = new Dictionary<List<double>, List<double>>();
            for (int i = 0; i < amountOfDataBatches; i++)
            {


                List<double> inputData = new List<double>();
                for (int j = 0; j < inputsAvaliable; j++)
                {

                    inputData.Add(randomGenerator.NextDouble());

                }

                syntheticData.Add(inputData, generateResults(inputData));
            }
            return syntheticData;

        }


        private static List<double> generateResults(List<double> inputData)
        {
            List<double> outputData = new List<double>();
            foreach (var input in inputData)
            {
                var output = Math.Tan(input) - 0.78; outputData.Add(output);

            }
            return outputData;
        }

        public static void NumericalGradCheck(Value weight, Action rebuildForwardPass, (List<double>,List<double>) sample, MLP mlp, double h = 1e-5)
        {
            // original gradient after 1 backwards run
            Console.WriteLine($"original gradient:{weight.Grad}");
            var OriginalWeight = weight.Data;
            weight.Data += h;
            rebuildForwardPass();
            var lossPlus = mlp.ComputeLoss(sample.Item2);
            weight.Data -= 2 * h;
            rebuildForwardPass();
            var lossMinus = mlp.ComputeLoss(sample.Item2);
            var numericalGrad = (lossPlus.Data - lossMinus.Data) / (2 * h);
            weight.Data = OriginalWeight;

            Console.WriteLine($"NumericalGrad = {numericalGrad}  ::  BackPropGrad = {weight.Grad}");
        }

    }
}
