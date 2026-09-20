using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation
{
    public class Value
    {
        public double Data { get; set; }
        public double Grad { get; set; } = 0.0;
        public string Operator { get; set; }
        public List<Value> Children = new List<Value>();
        public string Label { get; set; }

        public Action Backward;


        public Value(double data, string label, string operation = "")
        {
            this.Data = data;
            this.Operator = operation;
            this.Label = label;
        }

        public override string ToString()
        {
            var ChildrenString = "[";
            foreach (var v in Children)
            {
                if (!ChildrenString.EndsWith('['))
                { ChildrenString = ChildrenString + ','; }
                ChildrenString = ChildrenString + v.Label;

            };
            ChildrenString = ChildrenString + "]";

            return $"{{data: {this.Data.ToString()}, operation: {this.Operator}, Label: {this.Label}, Grad: {this.Grad} Children: {ChildrenString}}}";
        }
        public Value Add(Value other, string label)
        {
            Value val = new Value(this.Data + other.Data, label, "+");
            val.Children.Add(this);
            val.Children.Add(other);



            val.Backward = () =>
            {
                this.Grad += (1.0 * val.Grad);
                other.Grad += (1.0 * val.Grad);
            };
            return val;
        }

        public Value Multiply(Value other, string label)
        {
            Value val = new Value(this.Data * other.Data, label, "*");
            val.Children.Add(this);
            val.Children.Add(other);


            val.Backward = () =>
            {
                this.Grad += (other.Data * val.Grad);
                other.Grad += (this.Data * val.Grad);
            };
            return val;

        }
        
        public Value Divide(Value other, string label)
        {
            var outVal = new Value(this.Data / other.Data, label);
            outVal.Children.Add(this);
            outVal.Children.Add(other);
            outVal.Backward = () =>
            {
                this.Grad += (1.0 / other.Data) * outVal.Grad;
                other.Grad += (-this.Data / (other.Data * other.Data)) * outVal.Grad;
            };
            return outVal;
        }

        public Value Substract(Value other, string label)
        {
            Value val = new Value(this.Data - other.Data, label, "-");
            val.Children.Add(this);
            val.Children.Add(other);

            val.Backward = () =>
            {
                this.Grad += (1.0 * val.Grad);
                other.Grad -= (1.0 * val.Grad);
            };
            return val;
        }

        public Value tanh(string label)
        {
            Value val = new Value((Math.Exp(2 * this.Data) - 1) / (Math.Exp(2 * this.Data) + 1), label, "tanh");
            val.Children.Add(this);

            val.Backward = () =>
            {
                this.Grad += (1 - Math.Pow(val.Data, 2)) * val.Grad;

            };

            return val;
        }

        public Value LossCalc(Value expected, string label)
        {
            var data = Math.Pow(this.Data - expected.Data, 2);
            Value val = new Value(data, label);
            val.Children.Add(this);

            val.Backward = () =>
            {
                this.Grad += (2 * (this.Data - expected.Data)) * val.Grad;

            };

            return val;
        }

        public void AdjustWeight(double learningRate)
        {
            Data = Data - learningRate * Grad;
        }

        public void ResetGradients()
        {
            var topo = new List<Value>();
            var visited = new HashSet<Value>();

            void BuildTopo(Value v)
            {
                if (visited.Add(v))
                {
                    foreach (var child in v.Children)
                        BuildTopo(child);
                    topo.Add(v);
                }
            }

            BuildTopo(this);

            for (int i = topo.Count - 1; i >= 0; i--)
            {
                topo[i].Grad = 0.0;
            }
        }

        public void RunBackward()
        {
            var topo = new List<Value>();
            var visited = new HashSet<Value>();

            void BuildTopo(Value v)
            {
                if (visited.Add(v))
                {
                    foreach (var child in v.Children)
                        BuildTopo(child);
                    topo.Add(v);
                }
            }

            BuildTopo(this);

            this.Grad = 1.0;

            for (int i = topo.Count - 1; i >= 0; i--)
            {
                topo[i].Backward?.Invoke();
            }
        }
    }
}
