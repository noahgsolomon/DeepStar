from graphviz import Digraph

def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v.children:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name=uid, label=f"{{ data {n.data:.4f} | name {n.name} | grad {n.grad:.4f} }}", shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot
  
def visualize(model):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR', 'ranksep': '2'}) 

    # Add an input node
    dot.node('input', 'Input', shape='ellipse')

    # Iterate through each layer and add nodes and edges
    for i, layer in enumerate(model.layers):
        print(f'{i}, {layer}')
        with dot.subgraph(name=f'cluster_{i}') as c:
            c.attr(color='blue')
            c.node_attr['style'] = 'filled'
            c.node_attr['color'] = 'white'
            c.node_attr['shape'] = 'ellipse'

            # Label for the layer
            c.attr(label=f"Layer {i}\n{layer.__class__.__name__}\n" + (f"Activation: {layer.activation}" if layer.activation else ""))

            # Loop through neurons in the layer
            for j, neuron in enumerate(layer.neurons):
                neuron_name = f'neuron_{i}_{j}'

                # If it's the last layer and has only one neuron, treat it as the output
                if i == len(model.layers) - 1 and len(layer.neurons) == 1:
                    neuron_label = 'Output\n' + f'Bias: {neuron.b.data:.4f}'
                    dot.node(neuron_name, neuron_label, shape='ellipse')
                else:
                    neuron_label = f'Neuron {j}\nBias: {neuron.b.data:.4f}'
                    c.node(neuron_name, neuron_label, shape='ellipse')

                # Connect previous layer to neurons in the current layer
                if i == 0:  # First layer, connect from input
                    input_edge_label = f'w: {neuron.w[0].data:.4f}'
                    c.edge('input', neuron_name, label=input_edge_label)
                else:
                    for k, weight in enumerate(neuron.w):
                        # Connect neurons to the next layer's neurons
                        prev_neuron_name = f'neuron_{i-1}_{k}'
                        weight_edge_label = f'w: {weight.data:.4f}'
                        c.edge(prev_neuron_name, neuron_name, label=weight_edge_label)

    return dot