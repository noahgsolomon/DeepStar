import numpy as np
from nn import Embedding
from onnx import TensorProto # type: ignore
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor # type: ignore


def export_to_onnx(input_dim, output_dim, model, name='model'):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, input_dim])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, output_dim])

    nodes = []
    tensors = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, Embedding):
            embedding_values = [emb.data for emb in layer.embeddings.flatten()]
            embedding_tensor = make_tensor(f'Embedding{i+1}', TensorProto.FLOAT, [layer.num_embeddings, layer.embedding_dim], embedding_values)
            tensors.append(embedding_tensor)

            X_name = 'X' if i == 0 else f'X{i}'
            Y_name = f'X{i+1}'
            Y_gather_name = f'Gather{Y_name}'

            cast_node_name = f'CastToINT64_{i}'
            nodes.append(make_node('Cast', [X_name], [cast_node_name], to=TensorProto.INT64))

            # Update Gather input to use the casted indices
            nodes.append(make_node('Gather', [f'Embedding{i+1}', cast_node_name], [Y_gather_name]))

            nodes.append(make_node('Flatten', [Y_gather_name], [Y_name]))

        else:
            W_values = np.array([[0.] * len(layer.neurons) for _ in range(len(layer.neurons[0].w))], dtype=np.float32)
            for j in range(len(layer.neurons[0].w)):
                for k in range(len(layer.neurons)):
                        W_values[j][k] = (layer.neurons[k].w[j].data)

            B_values = np.array([])
            for n in layer.neurons:
                B_values = np.append(B_values, n.b.data)

            W = make_tensor(f'W{i+1}', TensorProto.FLOAT, [layer.nin, layer.nout], W_values)
            B = make_tensor(f'B{i+1}', TensorProto.FLOAT, [layer.nout], B_values)

            tensors.extend([W, B])

            X_name = 'X' if i == 0 else f'X{i}'
            Y_name = f'X{i+1}'

            nodes.append(make_node('MatMul', [X_name, W.name], [f'{X_name}W']))
            nodes.append(make_node('Add', [f'{X_name}W', B.name], [f'{X_name}WB' if layer.activation else Y_name]))

            if layer.activation:
                nodes.append(make_node(layer.activation, [f'{X_name}WB'], [Y_name]))

    nodes[-1].output[0] = 'Y'

    graph = make_graph(nodes, name, [X], [Y], tensors)
    onnx_model = make_model(graph, producer_name='star-model')

    return onnx_model
