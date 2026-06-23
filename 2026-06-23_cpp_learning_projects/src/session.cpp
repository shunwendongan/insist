#include "minirpc/session.hpp"

#include "minirpc/framing.hpp"
#include "minirpc/protobuf_wire.hpp"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

namespace minirpc {

std::atomic_uint64_t Session::next_id_{1};

Session::Session(EventLoop& loop, int socket_fd)
    : loop_(loop),
      socket_fd_(socket_fd),
      id_(next_id_.fetch_add(1, std::memory_order_relaxed)) {
}

Session::~Session() {
    if (socket_fd_ != -1) {
        ::close(socket_fd_);
        socket_fd_ = -1;
    }
}

void Session::start() {
    set_non_blocking(socket_fd_);
    auto self = shared_from_this();
    loop_.add(socket_fd_, EPOLLIN | EPOLLRDHUP, [self](std::uint32_t events) {
        self->handle_events(events);
    });
    registered_ = true;
}

void Session::send(RpcMessage message, WriteHandler handler) {
    const auto payload = ProtobufWireCodec::encode(message);
    if (payload.size() > kMaxFrameSize) {
        if (handler) {
            handler(std::make_error_code(std::errc::message_size));
        }
        return;
    }

    auto frame = make_frame(payload);
    auto self = shared_from_this();
    loop_.post([self, frame = std::move(frame), handler = std::move(handler)]() mutable {
        if (!self->open_) {
            if (handler) {
                handler(std::make_error_code(std::errc::not_connected));
            }
            return;
        }
        self->write_queue_.push_back(OutgoingFrame{std::move(frame), 0, std::move(handler)});
        self->update_events();
        self->async_write();
    });
}

void Session::close() {
    auto self = shared_from_this();
    loop_.post([self] { self->close_from_loop(); });
}

void Session::set_message_handler(MessageHandler handler) {
    message_handler_ = std::move(handler);
}

void Session::set_close_handler(CloseHandler handler) {
    close_handler_ = std::move(handler);
}

int Session::fd() const {
    return socket_fd_;
}

std::uint64_t Session::id() const {
    return id_;
}

bool Session::is_open() const {
    return open_;
}

void Session::handle_events(std::uint32_t events) {
    if (!open_) {
        return;
    }

    if ((events & (EPOLLERR | EPOLLHUP)) != 0) {
        close_from_loop();
        return;
    }

    if ((events & EPOLLIN) != 0) {
        async_read();
    }

    if (open_ && (events & EPOLLOUT) != 0) {
        async_write();
    }

    if (open_ && (events & EPOLLRDHUP) != 0) {
        close_from_loop();
    }
}

void Session::async_read() {
    char chunk[8192];

    while (open_) {
        const auto n = ::recv(socket_fd_, chunk, sizeof(chunk), 0);
        if (n > 0) {
            read_buffer_.insert(read_buffer_.end(), chunk, chunk + n);
            continue;
        }
        if (n == 0) {
            close_from_loop();
            return;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        fail_and_close(std::error_code(errno, std::generic_category()));
        return;
    }

    while (open_) {
        std::string payload;
        const auto result = try_pop_frame(read_buffer_, payload);
        if (result == FrameReadResult::NeedMore) {
            break;
        }
        if (result == FrameReadResult::TooLarge) {
            fail_and_close(std::make_error_code(std::errc::message_size));
            return;
        }

        auto decoded = ProtobufWireCodec::decode(payload);
        if (!decoded) {
            fail_and_close(std::make_error_code(std::errc::protocol_error));
            return;
        }

        if (message_handler_) {
            message_handler_(shared_from_this(), *decoded);
        }
    }
}

void Session::async_write() {
    while (open_ && !write_queue_.empty()) {
        auto& frame = write_queue_.front();
        const char* data = frame.bytes.data() + static_cast<std::ptrdiff_t>(frame.written);
        const auto remaining = frame.bytes.size() - frame.written;
        const auto n = ::send(socket_fd_, data, remaining, MSG_NOSIGNAL);

        if (n > 0) {
            frame.written += static_cast<std::size_t>(n);
            if (frame.written == frame.bytes.size()) {
                auto handler = std::move(frame.handler);
                write_queue_.pop_front();
                if (handler) {
                    handler({});
                }
            }
            continue;
        }

        if (n == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            break;
        }
        if (n == -1 && errno == EINTR) {
            continue;
        }

        fail_and_close(std::error_code(errno, std::generic_category()));
        return;
    }

    if (open_) {
        update_events();
    }
}

void Session::update_events() {
    if (!registered_ || !open_) {
        return;
    }

    std::uint32_t events = EPOLLIN | EPOLLRDHUP;
    if (!write_queue_.empty()) {
        events |= EPOLLOUT;
    }
    loop_.modify(socket_fd_, events);
}

void Session::close_from_loop() {
    if (!open_) {
        return;
    }

    open_ = false;
    const auto saved_fd = socket_fd_;
    socket_fd_ = -1;

    if (registered_) {
        loop_.remove(saved_fd);
        registered_ = false;
    }

    if (saved_fd != -1) {
        ::close(saved_fd);
    }

    while (!write_queue_.empty()) {
        auto handler = std::move(write_queue_.front().handler);
        write_queue_.pop_front();
        if (handler) {
            handler(std::make_error_code(std::errc::not_connected));
        }
    }

    if (close_handler_) {
        close_handler_(shared_from_this());
    }
}

void Session::fail_and_close(const std::error_code& error) {
    while (!write_queue_.empty()) {
        auto handler = std::move(write_queue_.front().handler);
        write_queue_.pop_front();
        if (handler) {
            handler(error);
        }
    }
    close_from_loop();
}

}  // namespace minirpc

