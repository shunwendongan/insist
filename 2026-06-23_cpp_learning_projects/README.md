# Mini RPC Protobuf Async

一个用于学习 C++ 网络编程的 mini RPC 项目。它使用 TCP 长连接、protobuf wire-format 消息、异步读写、`shared_ptr` 生命周期管理和 handler 回调机制，演示客户端与服务端双端全通信。

## 特性

- C++17，无第三方运行时依赖。
- `proto/rpc.proto` 定义 RPC 消息协议。
- 内置轻量 `ProtobufWireCodec`，可在没有 `protoc` 的环境中直接编译。
- 4 字节大端长度头处理 TCP 粘包/半包。
- `EventLoop` 基于 Linux `epoll + eventfd`。
- `Session` 使用 `std::enable_shared_from_this`，异步回调捕获 `shared_ptr` 保活。
- `Session::send` 异步写队列支持跨线程投递。
- `MessageHandler / WriteHandler / CloseHandler` 分离网络层和业务层。
- 示例服务支持 `DemoService.echo`、`DemoService.upper`、`DemoService.time`。

## 构建

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 运行示例

终端 1 启动服务端：

```bash
./build/minirpc_server 5555
```

终端 2 启动客户端：

```bash
./build/minirpc_client 127.0.0.1 5555
```

客户端会异步发送三个 RPC 请求和一个 `NOTIFY`，服务端会在连接建立时主动推送 welcome 通知，随后返回对应响应。

## 项目结构

```text
.
├── include/minirpc       # 公共头文件
├── src                   # RPC 核心实现
├── examples              # 服务端和客户端示例
├── proto/rpc.proto       # protobuf 协议定义
├── tests                 # 协议与 frame 单测
└── docs                  # 技术文档、流程图、架构图
```

## 文档

- 技术文档：[docs/technical_design.md](docs/technical_design.md)
- 架构图 Mermaid 源文件：[docs/architecture.mmd](docs/architecture.mmd)
- 流程图 Mermaid 源文件：[docs/flowchart.mmd](docs/flowchart.mmd)

## 设计要点

`Session` 是连接对象，也是异步读写的核心。注册到 `EventLoop` 时捕获 `shared_from_this()`，确保回调执行期间对象不会被提前析构。

```cpp
auto self = shared_from_this();
loop_.add(socket_fd_, EPOLLIN | EPOLLRDHUP, [self](std::uint32_t events) {
    self->handle_events(events);
});
```

读流程：`EPOLLIN -> recv -> frame 解析 -> protobuf 解码 -> MessageHandler`。

写流程：`Session::send -> protobuf 编码 -> frame 打包 -> write_queue_ -> EPOLLOUT -> async_write -> WriteHandler`。

## 后续扩展

- 替换为官方 Protobuf 生成代码。
- 增加 RPC 服务注册中心。
- 增加超时、重试、心跳、连接池。
- 增加业务线程池，避免阻塞 I/O 线程。
- 支持 TLS、鉴权、压缩和链路追踪。

