#include "minirpc/server.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <string>
#include <system_error>
#include <utility>

namespace {

std::string now_text() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::string text = std::ctime(&time);
    if (!text.empty() && text.back() == '\n') {
        text.pop_back();
    }
    return text;
}

minirpc::RpcMessage handle_demo_request(const minirpc::RpcMessage& request) {
    minirpc::RpcMessage response;
    response.id = request.id;
    response.kind = minirpc::RpcMessage::Kind::Response;
    response.service = request.service;
    response.method = request.method;

    if (request.service != "DemoService") {
        response.status = 404;
        response.error = "unknown service: " + request.service;
        return response;
    }

    if (request.method == "echo") {
        response.payload = request.payload;
    } else if (request.method == "upper") {
        response.payload = request.payload;
        std::transform(response.payload.begin(), response.payload.end(), response.payload.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });
    } else if (request.method == "time") {
        response.payload = now_text();
    } else {
        response.status = 404;
        response.error = "unknown method: " + request.method;
    }

    return response;
}

}  // namespace

int main(int argc, char** argv) {
    const auto port = static_cast<std::uint16_t>(argc > 1 ? std::stoi(argv[1]) : 5555);

    minirpc::EventLoop loop;
    minirpc::TcpServer server(loop, "0.0.0.0", port);

    server.set_session_handler([](const minirpc::Session::Ptr& session) {
        std::cout << "[server] session #" << session->id() << " connected\n";

        minirpc::RpcMessage welcome;
        welcome.kind = minirpc::RpcMessage::Kind::Notify;
        welcome.service = "System";
        welcome.method = "welcome";
        welcome.payload = "welcome, session #" + std::to_string(session->id());
        session->send(std::move(welcome));
    });

    server.set_message_handler([](const minirpc::Session::Ptr& session, const minirpc::RpcMessage& message) {
        std::cout << "[server] recv " << minirpc::to_string(message.kind)
                  << " id=" << message.id
                  << " " << message.service << "." << message.method
                  << " payload=\"" << message.payload << "\"\n";

        if (message.kind == minirpc::RpcMessage::Kind::Request) {
            session->send(handle_demo_request(message), [](const std::error_code& error) {
                if (error) {
                    std::cerr << "[server] async write failed: " << error.message() << '\n';
                }
            });
        }
    });

    server.start();
    std::cout << "[server] listening on 0.0.0.0:" << port << '\n';
    loop.run();
    return 0;
}
