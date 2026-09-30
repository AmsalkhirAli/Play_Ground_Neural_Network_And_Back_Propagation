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

    }
}
