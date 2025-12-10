# CuTile IR Dump 库

### 使用

```python
import cuda.tile as ct
from ir_dump import CutileIrDump

# 定义 kernel
@ct.kernel
def softmax(x, out, r: ct.Constant):
    c = ct.load(x, (0, ct.bid(0)), (512, r))
    max = ct.max(c, axis=0, keepdims=True)
    num = ct.exp(c - max)
    den = ct.sum(num, axis=0, keepdims=True)
    smax = num / den
    ct.store(out, (0, ct.bid(0)), smax)

# 创建 dumper
dumper = CutileIrDump(output_dir="./ir_artifacts")

# 创建 mock tensors
x = dumper.create_mock_tensor((512, 128), dtype="float32")
out = dumper.create_mock_tensor((512, 128), dtype="float32")

# 编译并导出 IR
files = dumper.compile_kernel(
    softmax,
    args=[x, out, 8],
    kernel_name="softmax"
)

print(f"生成的文件: {files}")
# 输出: {'cutileir': './ir_artifacts/softmax.cutileir', 
#        'bytecode': './ir_artifacts/softmax.cutile'}
```

### 获取 IR 字符串

```python
# 不写入文件，直接获取 IR 字符串
ir_string = dumper.dump_ir_to_string(
    softmax,
    args=[x, out, 8],
    ir_type="cutileir"
)
print(ir_string)
```

### 从环境变量创建

```python
import os

# 设置环境变量
os.environ["CUDA_TILE_DUMP_TILEIR"] = "./ir_artifacts"
os.environ["CUDA_TILE_DUMP_BYTECODE"] = "./ir_artifacts"

# 从环境变量创建
dumper = CutileIrDump.from_env()
```

### 链式调用

```python
dumper = (CutileIrDump(output_dir="./ir_artifacts")
          .set_compiler_options(opt_level=3, num_ctas=None))
```

## API 文档

### CutileIrDump

#### `__init__(output_dir, dump_cutileir, dump_bytecode, dump_mlir, compute_capability)`

初始化 IR Dump 工具。

**参数：**
- `output_dir` (str): IR 文件输出目录，默认 `"./ir_dump"`
- `dump_cutileir` (bool): 是否导出 CuTile IR (.cutileir)，默认 `True`
- `dump_bytecode` (bool): 是否导出 Bytecode (.cutile)，默认 `True`
- `dump_mlir` (bool): 是否导出 MLIR (.cuda_tile.mlir)，默认 `False`
- `compute_capability` (Tuple[int, int]): GPU 计算能力，如 `(8, 9)`，`None` 则自动检测

#### `compile_kernel(kernel_func, args, constants, grid, block, kernel_name)`

编译 kernel 并导出 IR。

**参数：**
- `kernel_func`: 使用 `@ct.kernel` 装饰的 kernel 函数
- `args` (List[Any]): kernel 参数列表（可以是 MockTensor 或真实 tensor）
- `constants` (Dict[str, Any]): 常量参数字典（可选）
- `grid` (Tuple[int, ...]): grid 维度（可选）
- `block` (Tuple[int, ...]): block 维度（可选）
- `kernel_name` (str): 导出的文件名前缀，默认使用 kernel 函数名

**返回：**
- `Dict[str, str]`: 生成的文件路径字典

#### `create_mock_tensor(shape, dtype)`

创建用于类型推断的 Mock Tensor。

**参数：**
- `shape` (Tuple[int, ...]): tensor 形状
- `dtype` (str): 数据类型，如 `"float32"`, `"float16"`, `"int32"`

**返回：**
- `MockTensor`: MockTensor 对象

#### `dump_ir_to_string(kernel_func, args, constants, ir_type)`

编译 kernel 并返回 IR 字符串（不写入文件）。

**参数：**
- `kernel_func`: kernel 函数
- `args` (List[Any]): 参数列表
- `constants` (Dict[str, Any]): 常量参数（可选）
- `ir_type` (str): IR 类型，可选 `"cutileir"`, `"bytecode"`, `"mlir"`

**返回：**
- `str`: IR 的字符串表示

#### `from_env()` (静态方法)

从环境变量创建 CutileIrDump 实例。

**环境变量：**
- `CUDA_TILE_DUMP_TILEIR`: CuTile IR 输出目录
- `CUDA_TILE_DUMP_BYTECODE`: Bytecode 输出目录
- `CUDA_TILE_DUMP_MLIR`: MLIR 输出目录

**返回：**
- `CutileIrDump`: 配置好的实例

#### `set_compiler_options(**options)`

设置编译器选项（链式调用）。

**参数：**
- `**options`: 编译器选项，传递给 `CompilerOptions`

**返回：**
- `CutileIrDump`: self，支持链式调用

### MockTensor

#### `__init__(shape, dtype_str)`

创建 MockTensor。

**参数：**
- `shape` (Tuple[int, ...]): tensor 形状
- `dtype_str` (str): 数据类型，默认 `"float32"`

## 示例

项目根目录下提供了两个完整的示例：

- [`softmax_ir_example.py`](../softmax_ir_example.py): Softmax kernel IR dump 示例
- [`attention_ir_example.py`](../attention_ir_example.py): Flash Attention kernel IR dump 示例

运行示例：

```bash
# Softmax 示例
python softmax_ir_example.py

# Flash Attention 示例
python attention_ir_example.py
```

## 输出文件格式

### CuTile IR (.cutileir)

高级 IR 表示，包含完整的类型信息和操作语义。

```
func @softmax(x: Array[float32,(?,?):(128,1)], out: Array[float32
,(?,?):(128,1)], r.0: int32):
    $token: Token = make_token()
    $5: const int32 = typed_const(value=0)
    ...
```

### Bytecode (.cutile)

序列化的 IR，二进制格式，可用于后续编译和优化。
