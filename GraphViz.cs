using System;
using System.Collections.Generic;
using System.Text;

namespace BackPropagation
{

    public static class GraphViz
    {
        public static string ToDot(Value root)
        {
            var nodes = new HashSet<Value>();
            var edges = new HashSet<(Value, Value)>();

            void Trace(Value v)
            {
                if (nodes.Add(v))
                {
                    foreach (var child in v.Children)
                    {
                        edges.Add((child, v));
                        Trace(child);
                    }
                }
            }
            Trace(root);

            var sb = new StringBuilder();
            sb.AppendLine("digraph G {");
            sb.AppendLine("  rankdir=LR;");

            foreach (var n in nodes)
            {
                string uid = n.GetHashCode().ToString();

                // the value node itself
                sb.AppendLine($"  \"{uid}\" [label=\"{{ {n.Label} | data {n.Data:F4} | grad {n.Grad:F4} }}\", shape=record];");

                // if it came from an operation, add a small op-node feeding into it
                if (!string.IsNullOrEmpty(n.Operator))
                {
                    string opUid = uid + n.Operator;
                    sb.AppendLine($"  \"{opUid}\" [label=\"{n.Operator}\"];");
                    sb.AppendLine($"  \"{opUid}\" -> \"{uid}\";");
                }
            }

            foreach (var (from, to) in edges)
            {
                string fromUid = from.GetHashCode().ToString();
                string toOpUid = to.GetHashCode() + to.Operator;
                sb.AppendLine($"  \"{fromUid}\" -> \"{toOpUid}\";");
            }

            sb.AppendLine("}");
            return sb.ToString();
        }

        public static void PrintTree(Value v, string indent = "")
        {
            Console.WriteLine($"{indent}{v.Label} = {v.Data:F4} (grad {v.Grad:F4}) [{v.Operator}]");
            foreach (var child in v.Children)
                PrintTree(child, indent + "  ");
        }
    }
}

