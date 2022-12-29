import torch
import torchvision
import numpy as np
import torch._dynamo
from typing import List

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

from torchvision.models import resnet18
def init_model():
    return resnet18().to(torch.float32).cuda()

def evaluate(mod, inp):
    return mod(inp)

def time_compare():
    model = init_model()
    evaluate_opt = torch.compile(evaluate, mode="reduce-overhead")
    print("~" * 10)
    eager_times = []
    compile_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        _, eager_time = timed(lambda: evaluate(model, inp))
        eager_times.append(eager_time)
        print(f"eager eval time {i}: {eager_time}")

    print("~" * 10)

    compile_times = []
    for i in range(N_ITERS):
        inp = generate_data(16)[0]
        _, compile_time = timed(lambda: evaluate_opt(model, inp))
        compile_times.append(compile_time)
        print(f"compile eval time {i}: {compile_time}")
    print("~" * 10)

    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print("~" * 10)


"""
Testing torch.compile speed up effect in resnet18
"""
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

def test_torch_compile():
    model = init_model()
    time_default = []
    for i in range(20):
        _, time = timed(lambda:model(generate_data(100)[0]))
        print(f"time {i}:{time}")
        time_default.append(time)
    print("-----------")
    time_compile = []
    opt_model0 = torch.compile(model, mode="default")
    for i in range(20):
        _, time = timed(lambda:opt_model0(generate_data(100)[0]))
        print(f"time {i}:{time}")
        time_compile.append(time)
    print("-----------")
    torch._dynamo.reset()
    opt_model1 = torch.compile(model, mode="reduce-overhead")
    time_compile_mode_reduce = []
    for i in range(20):
        _, time = timed(lambda:opt_model1(generate_data(100)[0]))
        print(f"time {i}:{time}")
        time_compile_mode_reduce.append(time)
    print("-----------")
    torch._dynamo.reset()
    opt_model3 = torch.compile(model, backend=custom_backend)
    time_compile_backend = []
    for i in range(20):
        _, time = timed(lambda:opt_model3(generate_data(100)[0]))
        print(f"time {i}:{time}")
        time_compile_backend.append(time)
    
    print("-----------")
    time_default_med = np.median(time_default)
    time_compile_med = np.median(time_compile)
    time_compile_mode_reduce_med = np.median(time_compile_mode_reduce)
    time_compile_backend_med = np.median(time_compile_backend)

    print(f"default median time:{time_default_med}")
    print(f"under torch.compile: {time_compile_med}")
    print(f"under reduce-overhead mode: {time_compile_mode_reduce_med}")
    print(f"under custom backend: {time_compile_backend_med}")
    print("----------------")


"""
Another test for further comparing
"""

def test_without_compile():   
    model = init_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    times = []
    for i in range(10):
        x = torch.randn(16, 3, 224, 224).cuda()
        optimizer.zero_grad()
        out, time = timed(lambda: model(x))
        times.append(time)
        print(time)
        out.sum().backward()
        optimizer.step()
    return np.median(times)
    
def test_with_compile():   
    model = init_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    compiled_model = torch.compile(model)
    times = []
    for i in range(10):
        x = torch.randn(16, 3, 224, 224).cuda()
        optimizer.zero_grad()
        out, time = timed(lambda: compiled_model(x))
        print(time)
        times.append(time)
        out.sum().backward()
        optimizer.step()
    return np.median(times)

def test3_time():
    torch._dynamo.reset()
    print("--------time without torch.compile--------")
    median_time1 = test_without_compile()
    print(f"median time: {median_time1}")
    print("--------time with torch.compile--------")
    torch._dynamo.reset()
    median_time2 = test_with_compile()
    print(f"median time: {median_time2}")
    speedup = median_time1 / median_time2
    print(f"speed-up: {speedup}")
    return speedup

def test3():
    speedups = []
    for i in range(10):
        speedup = test2_time()
        speedups.append(speedup)
    print("-----------")
    print(f"final speed-up res: {speedups}")
    speedup_average = np.mean(speedups)
    print(f"average speedup: {speedup_average}")

time_compare()
test_torch_compile()
test3()

