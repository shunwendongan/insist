#pragma once

#include "minirpc/event_loop.hpp"
#include "minirpc/session.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace minirpc {

class TcpServer {
public:
    using SessionHandler = std::function<void(const Session::Ptr&)>;
    using MessageHandler = Session::MessageHandler;

    TcpServer(EventLoop& loop, std::string host, std::uint16_t port);
    ~TcpServer();

    TcpServer(const TcpServer&) = delete;
    TcpServer& operator=(const TcpServer&) = delete;

    void start();
    void stop();

    void set_session_handler(SessionHandler handler);
    void set_message_handler(MessageHandler handler);

private:
    void bind_and_listen();
    void accept_loop();
    void close_listener();

    EventLoop& loop_;
    std::string host_;
    std::uint16_t port_ = 0;
    int listen_fd_ = -1;
    bool started_ = false;

    SessionHandler session_handler_;
    MessageHandler message_handler_;
    std::unordered_map<std::uint64_t, Session::Ptr> sessions_;
};

}  // namespace minirpc

