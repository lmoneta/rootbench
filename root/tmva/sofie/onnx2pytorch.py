import os
import torch
import onnx2torch

import onnx


def get_input_shape(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Get the graph from the model
    graph = model.graph

    # Loop through all the inputs and get their shapes
    input_shapes = {}
    for input_tensor in graph.input:
        # The shape is stored in the type information of the tensor
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        input_shapes[input_tensor.name] = shape

    return input_shapes


def convert(path_onnx, path_torch):
    model = onnx2torch.convert(path_onnx)
    model = model.eval()
    model = torch.jit.script(model)
    torch.jit.save(model, path_torch)


def convert_directory(dir):
    for filename in os.listdir(dir):
        if filename.endswith(".onnx"):
            path_onnx = os.path.join(dir, filename)

            path_save = f"{filename[:-5]}.pt"
            path_torch = os.path.join(dir, path_save)

            # print(f"Converting {path_onnx} to {path_torch}")
            try:
                convert(path_onnx, path_torch)

                input_shapes = get_input_shape(path_onnx)
                print(f"Input shapes {input_shapes}, {os.path.basename(path_torch)}")

            except Exception as e:
                # print(f"Failed to convert {path_onnx} to {path_torch}")
                # print(e)
                pass


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory containing ONNX models",
        default=".",
    )
    args = parser.parse_args()

    convert_directory(args.dir)


if __name__ == "__main__":
    main()
