## CUDA 高效策略

### 高效公式

* 最大化计算强度

$$
\frac{Math}{Memory}
$$
其中 $Math$ 指数学计算量，$Memory$ 指每个线程的内存。

即要求：
1. 最大化每个线程的计算量
2. 最小化每个线程的内存读取速度
    * 每个线程读取的数据量少
    * 每个线程读取的速度快(1. 本地内存 > 共享内存 >> 全局内存；2. 合并全局内存)

### 避免线程发散

**避免线程发散**：同一个线程块中的线程执行不同内容的代码，如：

1. kernel 中做条件判断
```cpp
__global__ void kernel() {
    if (/* condition */) {
        // some code
    } else {
        // some other code
    }
}
```

2. 循环长度不一

```cpp
__global__ void kernel() {
    // pre loop code
    for (int i = 0; i < threadIdx.x; i++) {
        // loop code
    }
    // post loop code
}
```

## kernel 加载方式

机器参数获取：`deviceQuery`。

kernel 的加载中，自定义的线程数、线程块数量等都不要超过系统本身的限制。否则会影响运行效率。

## 内存代码使用

### 本地变量

```cpp
// using local memory
__global__ void use_local_memory_GPU(float in) {
    float f;  // "f" is in local memory and private to each thread
    f = in;   // parameter "in" is in local memory and private to each thread
}

// First, call a kernel that shows using local memory
use_local_memory_GPU<<<1, 128>>>(2.0f);
```

### 全局变量

```cpp
__global__ void use_global_memory_GPU(float *array) {
    // 一般用指针，array 指针变量本身是 local memory 的
    // "array" is a pointer into global memory on the device
    array[threadIdx.x] = 2.0f * (float)threadIdx.x;
}

float *d_arr;
// allocate global memory on the device
cudaMalloc((void **)&d_arr, sizeof(float) * 128);
```

### 共享变量

```cpp
__global__ void use_shared_memory_GPU(float *array) {
    int i;
    int index = threadIdx.x;

}
```

## CUDA 同步

### 原子操作

原子操作解决的问题：对于很多线程需要同时读取或写入相同内存时，保证同一时间只有一个线程能进行操作。

原子操作：
* 只支持某些运算(+/-/min/nor，不支持求余或幂)和数据类型(整型)
* 运行顺序不定
* 安排不当，速度很慢(因为内部是串行的)

### __syncthreads()

* 线程块内线程同步
* 保证线程块内所有线程都执行到统一位置

### cudaStreamSynchronize()/cudaEventSynchronize()

* 主机端代码中使用 `cudaTreadSynchronize()` 实现 CPU 和 GPU 的线程同步
* kernel 启动后控制权将异步返回，利用该函数可以确定所有设备端线程均已运行结束