import importlib
from typing import Any

layers = [{
        "name": "Linear",
        "inputs": [
            {
                "name": "in_features",
                "description": "Number of input features. Specifies the dimensionality of the input data."
            },
            {
                "name": "out_features",
                "description": "Number of output features. Determines the dimensionality of the output data."
            }
        ]
    }, {
        "name": "Conv1d",
        "inputs": [
            {
                "name": "in_channels",
                "description": "Number of channels in the input image."
            },
            {
                "name": "out_channels",
                "description": "Number of channels produced by the convolution."
            },
            {
                "name": "kernel_size",
                "description": "Size of the convolving kernel."
            }
        ]
    }, {
        "name": "Conv2d",
        "inputs": [
            {
                "name": "in_channels",
                "description": "Number of channels in the input image."
            },
            {
                "name": "out_channels",
                "description": "Number of channels produced by the convolution."
            },
            {
                "name": "kernel_size",
                "description": "Size of the convolving kernel."
            }
        ]
    }, {
        "name": "Conv3d",
        "inputs": [
            {
                "name": "in_channels",
                "description": "Number of channels in the input volume."
            },
            {
                "name": "out_channels",
                "description": "Number of channels produced by the convolution."
            },
            {
                "name": "kernel_size",
                "description": "Size of the convolving kernel."
            }
        ]
    }, {
        "name": "RNN",
        "inputs": [
            {
                "name": "input_size",
                "description": "The number of expected features in the input."
            },
            {
                "name": "hidden_size",
                "description": "The number of features in the hidden state."
            }
        ]
    }, {
        "name": "LSTM",
        "inputs": [
            {
                "name": "input_size",
                "description": "The number of expected features in the input."
            },
            {
                "name": "hidden_size",
                "description": "The number of features in the hidden state."
            }
        ]
    }, {
        "name": "GRU",
        "inputs": [
            {
                "name": "input_size",
                "description": "The number of expected features in the input."
            },
            {
                "name": "hidden_size",
                "description": "The number of features in the hidden state."
            }
        ]
    }, {
        "name": "Dropout",
        "inputs": [
            {
                "name": "p",
                "description": "Probability of an element to be zeroed."
            }
        ]
    },
    {
        "name": "Dropout2d",
        "inputs": [
            {
                "name": "p",
                "description": "Probability of an element to be zeroed."
            }
        ]
    }, {
        "name": "Dropout3d",
        "inputs": [
            {
                "name": "p",
                "description": "Probability of an element to be zeroed."
            }
        ]
    }, {
        "name": "Embedding",
        "inputs": [
            {
                "name": "num_embeddings",
                "description": "Size of the dictionary of embeddings."
            },
            {
                "name": "embedding_dim",
                "description": "The size of each embedding vector."
            }
        ],
    }
]

activation_functions = [
    {
        "name": "ReLU",
        "description": "Applies the rectified linear unit function element-wise."
    }, {
        "name": "Softmax",
        "description": "Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1."
    }, {
        "name": "Tanh",
        "description": "Applies the Hyperbolic Tangent (Tanh) function element-wise."
    }
]


def get_layer(name: str, params: dict[str, Any]):
    nn = importlib.import_module("torch.nn")
    layer = getattr(nn, name)
    return repr(layer(**params))


def get_activation_function(name: str):
    nn = importlib.import_module("torch.nn")
    activation_function = getattr(nn, name)
    return repr(activation_function())
