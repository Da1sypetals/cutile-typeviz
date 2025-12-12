"""
MockTensor 类

用于创建模拟的 tensor 对象进行类型推断，不需要真实的 CUDA 设备
"""


class MockTensor:
    """模拟的 Tensor 对象，用于 cuda.tile kernel 的类型推断"""

    def __init__(self, shape, dtype="float32"):
        """
        创建 MockTensor

        Args:
            shape: tensor 形状，如 (512, 128)
            dtype_str: 数据类型，如 "float32", "float16", "int32"
        """
        self.shape = shape
        self.dtype_str = dtype
        self.device = "cuda"

        self.data_ptr = lambda: 0  # 模拟指针

        self.__mock_tensor_metadata__ = {
            "shape": shape,
            "dtype_str": dtype,
            "data": (0, False),  # (ptr, read_only)
            "version": 3,
        }

        # self.__cuda_array_interface__ = {
        #     "shape": shape,
        #     "typestr": dtype_map[dtype],
        #     "data": (0, False),  # (ptr, read_only)
        #     "version": 3,
        # }
