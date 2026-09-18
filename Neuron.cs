using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation
{
    class Neuron
    {
        public List<Value>Weights { get; set; }
        public Value output { get; set; }
        public Value Bias { get; set; }
        private Random r;

        public Neuron(int numberOfInputs)
        {
            r = new Random();
            Weights = new List<Value>();
            InitialiseWeightsForInputs(numberOfInputs);
            Bias = new Value(r.NextDouble(),"Bias");
        }

        private void InitialiseWeightsForInputs(int numberOfInputs)
        {
            for(int i=1; i<=numberOfInputs;i++)
            {
                var weight = new Value(r.NextDouble(), $"w{i}");
                Weights.Add(weight);

            }
        }

    }
}
