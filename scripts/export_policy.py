# TODO: Export trained policy for deployment.

def export(model_path, out_path):
    print(f"Exporting model from {model_path} to {out_path}")

if __name__ == "__main__":
    export("checkpoints/model.pt", "exported/model.onnx")
