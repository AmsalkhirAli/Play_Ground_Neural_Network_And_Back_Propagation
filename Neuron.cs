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
        private readonly Random _random;

        public Neuron()
        {
            _random= new Random();
            Weights = new List<Value>();
            Bias = new Value(0,"Bias");
        }

        public void InitialiseWeights(int numberOfInputs,bool isHiddenLayer = false)
        {
            InitialiseWeightsForInputs(numberOfInputs,isHiddenLayer);
        }

        private void InitialiseWeightsForInputs(int numberOfInputs,bool isHiddenLayer=false)
        {
            double limit= 0.0;

            if (isHiddenLayer)
            {
                limit = Math.Sqrt(6.0 / numberOfInputs);
            }
            else
            {
                limit = 1.0 / Math.Sqrt(numberOfInputs);
            }

            for (int i=1; i<=numberOfInputs;i++)
            {
                double randomValue = (_random.NextDouble() * 2 - 1) * limit; // range: [-limit, limit)
                var weight = new Value(randomValue, $"w{i}");
                Weights.Add(weight);
            }
        }
        
        public void Forward(List<Value> data,bool isHiddenlayer=false)
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
            if (isHiddenlayer)
            {
                v = v.Relu("");
            }
            else
            {
                v = v.tanh("");
            }
            Output = v;
        }

    }
}
