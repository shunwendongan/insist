#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <system_error>
#include <unordered_map>

namespace minirpc {

class EventLoop {
public:
    using EventHandler = std::function<void(std::uint32_t events)>;

    EventLoop();
    ~EventLoop();

    EventLoop(const EventLoop&) = delete;
    EventLoop& operator=(const EventLoop&) = delete;

    void add(int fd, std::uint32_t events, EventHandler handler);
    void modify(int fd, std::uint32_t events);
    void remove(int fd);

    void post(std::function<void()> task);
    void run();
    void stop();

private:
    void wakeup();
    void drain_wakeup();
    void drain_tasks();

    int epoll_fd_ = -1;
    int event_fd_ = -1;
    std::atomic_bool stopped_{false};

    std::mutex handlers_mutex_;
    std::unordered_map<int, EventHandler> handlers_;

    std::mutex tasks_mutex_;
    std::queue<std::function<void()>> tasks_;
};

void throw_system_error(const char* what);
void set_non_blocking(int fd);

}  // namespace minirpc

