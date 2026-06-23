#pragma once

#include "minirpc/event_loop.hpp"
#include "minirpc/session.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace minirpc {

class RpcClient {
public:
    using ResponseHandler = std::function<void(const RpcMessage&)>;
    using MessageHandler = Session::MessageHandler;

    explicit RpcClient(EventLoop& loop);
    ~RpcClient();

    RpcClient(const RpcClient&) = delete;
    RpcClient& operator=(const RpcClient&) = delete;

    void connect(const std::string& host, std::uint16_t port);
    void close();

    std::uint64_t call_async(std::string service,
                             std::string method,
                             std::string payload,
                             ResponseHandler handler);
    void notify(std::string service, std::string method, std::string payload);
    void set_message_handler(MessageHandler handler);

    bool connected() const;

private:
    void dispatch_message(const Session::Ptr& session, const RpcMessage& message);

    EventLoop& loop_;
    Session::Ptr session_;
    MessageHandler message_handler_;
    std::atomic_uint64_t next_request_id_{1};
    std::mutex pending_mutex_;
    std::unordered_map<std::uint64_t, ResponseHandler> pending_;
};

}  // namespace minirpc
