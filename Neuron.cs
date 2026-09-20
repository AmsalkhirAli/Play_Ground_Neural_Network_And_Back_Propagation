using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation
{
    public class Neuron
    {
        public List<Value>Weights { get; set; }
        public Value Output { get; set; }
        public Value Bias { get; set; }
        private Random Rand;
        public int NumberOfInputs { get; set; }

        public Neuron()
        {
            Rand = new Random();
            Weights = new List<Value>();
            Bias = new Value(Rand.NextDouble(),"Bias");
        }

        public void InitialiseWeights(int numberOfInputs)
        {
            InitialiseWeightsForInputs(numberOfInputs);
        }

        private void InitialiseWeightsForInputs(int numberOfInputs)
        {
            for(int i=1; i<=numberOfInputs;i++)
            {
                var weight = new Value(Rand.NextDouble(), $"w{i}");
                Weights.Add(weight);

            }
        }
        
        public Value Forward(List<Value> data)
        {

            List<Value> ForwardList = new List<Value>();
            for (int i = 0; i < data.Count; i++)
            {
               ForwardList.Add(data[i].Multiply(Weights[i],$"x{i+1}w{i+1}"));
            }

            Value v=null;
            foreach (var value in ForwardList)
            {
                if(v==null)
                    v = value;
                else v=v.Add(value,"");

            }

            v =  v.Add(Bias,"");
            v = v.tanh("");
            Output = v;
            return Output;

        }

    }
}
