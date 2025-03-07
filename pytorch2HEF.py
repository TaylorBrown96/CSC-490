#!/usr/bin/env python3
"""
pytorch_to_hef.py

Converts a PyTorch model to ONNX, then compiles it with the Hailo SDK to produce a .hef file.

Requirements:
  - PyTorch
  - Hailo SDK (with hailo_platform or hailo_sdk modules installed in Python)
  - Onnx & onnxruntime (for verifying exported ONNX, optional but recommended)

Usage:
  python pytorch_to_hef.py --model my_model.pth --hef_out my_model.hef
"""

import argparse
import os
import torch
import onnx
import onnxruntime
import numpy as np

# Hailo SDK modules. These names may differ depending on your SDK version:
try:
    from hailo_platform import HEF, PDKVersion, Compiler
except ImportError:
    raise ImportError("Could not import hailo_platform. Make sure Hailo SDK is installed and sourced.")


def load_pytorch_model(model_path, model_class=None):
    """
    Load a PyTorch model from a .pth or .pt file.
    - If you have a custom model class, instantiate it and load state_dict().
    - Otherwise, you might be using a standard torchvision model or a scripted model.
    """
    # Example if you have a custom class:
    if model_class is not None:
        model = model_class()
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        # If your model is a torch.jit.ScriptModule or you rely on tracing
        model = torch.jit.load(model_path, map_location='cpu')

    model.eval()
    return model


def export_to_onnx(model, input_shape, onnx_path, opset=11):
    """
    Export the PyTorch model to ONNX. 
    input_shape: e.g. (1, 3, 224, 224) for a single image, 3-channel, 224x224.
    """
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"[INFO] Exported model to ONNX: {onnx_path}")


def verify_onnx(onnx_path, input_shape):
    """
    Optional: run a forward pass with onnxruntime to ensure the ONNX is valid.
    """
    session = onnxruntime.InferenceSession(onnx_path)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    outputs = session.run(None, {"input": dummy_input})
    print("[INFO] ONNX Runtime output shape:", [o.shape for o in outputs])


def compile_onnx_to_hef(onnx_path, hef_path, target_pdk="hailo8"):
    """
    Use Hailo SDK's Compiler to compile an ONNX model into a .hef.
    
    - target_pdk can be "hailo8" or a specific PDK version (e.g. PDKVersion(x,y,z)).
      For example: pdk_version = PDKVersion(4,15,0)
    """
    compiler = Compiler()
    compiler.set_number_of_processes(os.cpu_count())

    # You can configure additional compiler settings here if needed, e.g. quantization range
    # compiler.set_quantization('per_tensor_symmetric') # example

    # Compile
    print(f"[INFO] Compiling {onnx_path} -> {hef_path} for {target_pdk} ...")
    hef = compiler.compile_onnx_model(onnx_path, target_device=target_pdk)
    hef.save(hef_path)
    print(f"[INFO] Compilation complete. HEF file saved to: {hef_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to HEF for Hailo.")
    parser.add_argument("--model", type=str, required=True, help="Path to the PyTorch model (.pth, .pt, or .jit).")
    parser.add_argument("--hef_out", type=str, default="output.hef", help="Desired output HEF file name.")
    parser.add_argument("--onnx_out", type=str, default="temp_model.onnx", help="Temporary ONNX file name.")
    parser.add_argument("--input_shape", type=str, default="1,3,224,224", 
                        help="Input shape for the model, e.g. 1,3,224,224")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    parser.add_argument("--skip_verify", action="store_true", help="Skip ONNX verification step.")
    args = parser.parse_args()

    # Parse input_shape "1,3,224,224" -> [1,3,224,224]
    shape_list = [int(x) for x in args.input_shape.split(",")]
    input_shape = tuple(shape_list)

    # 1. Load the PyTorch model
    #    If you have a custom class, replace `None` with your class, e.g. MyCustomModel
    model = load_pytorch_model(args.model, model_class=None)

    # 2. Export to ONNX
    export_to_onnx(model, input_shape, args.onnx_out, opset=args.opset)

    # 3. (Optional) Verify ONNX correctness with onnxruntime
    if not args.skip_verify:
        verify_onnx(args.onnx_out, input_shape)

    # 4. Compile ONNX -> HEF
    compile_onnx_to_hef(args.onnx_out, args.hef_out, target_pdk="hailo8")

    print("[INFO] Conversion to HEF completed successfully.")


if __name__ == "__main__":
    main()
