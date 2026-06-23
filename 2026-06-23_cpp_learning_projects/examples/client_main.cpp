#include "minirpc/client.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

namespace {

void print_response(const minirpc::RpcMessage& response) {
    if (response.status == 0) {
        std::cout << "[client] response id=" << response.id
                  << " payload=\"" << response.payload << "\"\n";
    } else {
        std::cout << "[client] response id=" << response.id
                  << " error=" << response.status << " " << response.error << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
    const auto port = static_cast<std::uint16_t>(argc > 2 ? std::stoi(argv[2]) : 5555);

    minirpc::EventLoop loop;
    minirpc::RpcClient client(loop);
    client.connect(host, port);

    client.set_message_handler([](const minirpc::Session::Ptr&, const minirpc::RpcMessage& message) {
        std::cout << "[client] recv " << minirpc::to_string(message.kind)
                  << " " << message.service << "." << message.method
                  << " payload=\"" << message.payload << "\"\n";
    });

    std::thread io_thread([&loop] {
        loop.run();
    });

    std::atomic_int remaining{3};
    auto on_response = [&](const minirpc::RpcMessage& response) {
        print_response(response);
        if (--remaining == 0) {
            client.close();
            loop.stop();
        }
    };

    client.notify("Client", "ready", "client can also push messages");
    client.call_async("DemoService", "echo", "hello mini rpc", on_response);
    client.call_async("DemoService", "upper", "full duplex protobuf", on_response);
    client.call_async("DemoService", "time", "", on_response);

    for (int i = 0; i < 30 && remaining > 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    client.close();
    loop.stop();
    io_thread.join();
    return remaining == 0 ? 0 : 1;
}
