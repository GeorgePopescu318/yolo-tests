import tensorrt as trt
import os

def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    if not os.path.exists(engine_file_path):
        raise FileNotFoundError(f"Engine file not found: {engine_file_path}")
    with open(engine_file_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise ValueError("Failed to deserialize the engine. Check if the engine file is valid and compatible with the TensorRT version.")
    return engine

# Replace with the path to your engine file
engine_path = "best130epics2ndDS.engine"
try:
    engine = load_engine(engine_path)
except Exception as e:
    print("Error loading engine:", e)
    exit(1)

fp16_bindings = 0
fp32_bindings = 0

# Iterate over all bindings (inputs and outputs)
for i in range(engine.num_bindings):
    binding_name = engine.get_binding_name(i)
    binding_dtype = engine.get_binding_dtype(i)
    print(f"Binding {i} ({binding_name}): {binding_dtype}")
    if binding_dtype == trt.DataType.HALF:
        fp16_bindings += 1
    elif binding_dtype == trt.DataType.FLOAT:
        fp32_bindings += 1

print("FP16 bindings count:", fp16_bindings)
print("FP32 bindings count:", fp32_bindings)

if fp16_bindings > 0:
    print("The engine is using FP16 precision for some bindings.")
else:
    print("The engine does not appear to be using FP16 precision for its bindings.")
