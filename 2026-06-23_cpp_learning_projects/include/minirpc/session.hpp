#pragma once

#include "minirpc/event_loop.hpp"
#include "minirpc/message.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <system_error>
#include <vector>
#include <atomic>

namespace minirpc {

class Session : public std::enable_shared_from_this<Session> {
public:
    using Ptr = std::shared_ptr<Session>;
    using MessageHandler = std::function<void(const Ptr&, const RpcMessage&)>;
    using CloseHandler = std::function<void(const Ptr&)>;
    using WriteHandler = std::function<void(const std::error_code&)>;

    Session(EventLoop& loop, int socket_fd);
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void start();
    void send(RpcMessage message, WriteHandler handler = {});
    void close();

    void set_message_handler(MessageHandler handler);
    void set_close_handler(CloseHandler handler);

    int fd() const;
    std::uint64_t id() const;
    bool is_open() const;

private:
    struct OutgoingFrame {
        std::string bytes;
        std::size_t written = 0;
        WriteHandler handler;
    };

    void handle_events(std::uint32_t events);
    void async_read();
    void async_write();
    void update_events();
    void close_from_loop();
    void fail_and_close(const std::error_code& error);

    EventLoop& loop_;
    int socket_fd_ = -1;
    std::uint64_t id_ = 0;
    bool open_ = true;
    bool registered_ = false;

    std::vector<char> read_buffer_;
    std::deque<OutgoingFrame> write_queue_;

    MessageHandler message_handler_;
    CloseHandler close_handler_;

    static std::atomic_uint64_t next_id_;
};

}  // namespace minirpc
