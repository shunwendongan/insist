#include "minirpc/client.hpp"

#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <stdexcept>
#include <system_error>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

namespace minirpc {

namespace {

int connect_tcp(const std::string& host, std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    const auto port_text = std::to_string(port);
    const int rc = ::getaddrinfo(host.c_str(), port_text.c_str(), &hints, &result);
    if (rc != 0) {
        throw std::runtime_error(std::string("getaddrinfo: ") + ::gai_strerror(rc));
    }

    int fd = -1;
    std::error_code last_error;

    for (addrinfo* item = result; item != nullptr; item = item->ai_next) {
        fd = ::socket(item->ai_family, item->ai_socktype | SOCK_CLOEXEC, item->ai_protocol);
        if (fd == -1) {
            last_error = std::error_code(errno, std::generic_category());
            continue;
        }

        if (::connect(fd, item->ai_addr, item->ai_addrlen) == 0) {
            break;
        }

        last_error = std::error_code(errno, std::generic_category());
        ::close(fd);
        fd = -1;
    }

    ::freeaddrinfo(result);

    if (fd == -1) {
        if (last_error) {
            throw std::system_error(last_error, "connect");
        }
        throw std::runtime_error("connect: no address resolved");
    }

    return fd;
}

}  // namespace

RpcClient::RpcClient(EventLoop& loop)
    : loop_(loop) {
}

RpcClient::~RpcClient() {
    close();
}

void RpcClient::connect(const std::string& host, std::uint16_t port) {
    if (session_ && session_->is_open()) {
        return;
    }

    const int fd = connect_tcp(host, port);
    session_ = std::make_shared<Session>(loop_, fd);
    session_->set_message_handler([this](const Session::Ptr& session, const RpcMessage& message) {
        dispatch_message(session, message);
    });
    session_->set_close_handler([this](const Session::Ptr&) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_.clear();
        session_.reset();
    });
    session_->start();
}

void RpcClient::close() {
    if (session_) {
        session_->close();
    }
}

std::uint64_t RpcClient::call_async(std::string service,
                                    std::string method,
                                    std::string payload,
                                    ResponseHandler handler) {
    if (!session_ || !session_->is_open()) {
        throw std::runtime_error("RpcClient is not connected");
    }

    const auto id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    RpcMessage message;
    message.id = id;
    message.kind = RpcMessage::Kind::Request;
    message.service = std::move(service);
    message.method = std::move(method);
    message.payload = std::move(payload);

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_[id] = std::move(handler);
    }
    session_->send(std::move(message), [this, id](const std::error_code& error) {
        if (error) {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            pending_.erase(id);
        }
    });
    return id;
}

void RpcClient::notify(std::string service, std::string method, std::string payload) {
    if (!session_ || !session_->is_open()) {
        throw std::runtime_error("RpcClient is not connected");
    }

    RpcMessage message;
    message.kind = RpcMessage::Kind::Notify;
    message.service = std::move(service);
    message.method = std::move(method);
    message.payload = std::move(payload);
    session_->send(std::move(message));
}

void RpcClient::set_message_handler(MessageHandler handler) {
    message_handler_ = std::move(handler);
}

bool RpcClient::connected() const {
    return session_ && session_->is_open();
}

void RpcClient::dispatch_message(const Session::Ptr& session, const RpcMessage& message) {
    if (message.kind == RpcMessage::Kind::Response) {
        ResponseHandler handler;
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);
            auto it = pending_.find(message.id);
            if (it != pending_.end()) {
                handler = std::move(it->second);
                pending_.erase(it);
            }
        }
        if (handler) {
            handler(message);
            return;
        }
    }

    if (message_handler_) {
        message_handler_(session, message);
    }
}

}  // namespace minirpc
