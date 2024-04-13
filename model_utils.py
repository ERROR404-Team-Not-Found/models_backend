import re
import importlib
from typing import Any

layers = [{
        "name": "Linear",
        "description": "Applies a linear transformation to the incoming data: y = x*A^2 + b.",
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
        "description": "Applies a 1D convolution over a quantized input signal composed of several quantized input planes.",
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
        "description": "Applies a 2D convolution over a quantized input signal composed of several quantized input planes.",
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
        "description": "Applies a 3D convolution over a quantized input signal composed of several quantized input planes.",
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
        "description": "Apply a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence. For each element in the input sequence.",
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
        "description": "A quantizable long short-term memory (LSTM).",
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
        "description": "Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.",
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
        "name": "Dropout1d",
        "description": "Randomly zero out entire channels for 1d values input.",
        "inputs": [
            {
                "name": "p",
                "description": "Probability of an element to be zeroed."
            }
        ]
    },
    {
        "name": "Dropout2d",
        "description": "Randomly zero out entire channels for 2d values input.",
        "inputs": [
            {
                "name": "p",
                "description": "Probability of an element to be zeroed."
            }
        ]
    }, {
        "name": "Dropout3d",
        "description": "Randomly zero out entire channels for 3d values input.",
        "inputs": [
            {
                "name": "p",
                "description": "Probability of an element to be zeroed."
            }
        ]
    }, {
        "name": "Embedding",
        "description": "A simple lookup table that stores embeddings of a fixed dictionary and size.",
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

log_pattern = re.compile(r'(?P<asctime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<levelname>\w+) - (?P<message>.*)')

# Function to parse a single log line
def parse_log_line(line):
    match = log_pattern.match(line)
    if match:
        return match.groupdict()
    return None


def read_log_file(file_path):
    entries = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parsed_line = parse_log_line(line)
                if parsed_line:
                    entries.append(parsed_line)
    except FileNotFoundError:
        print(f"Log file not found at {file_path}")
    except Exception as e:
        print(f"Failed to read log file: {str(e)}")
    return entries
