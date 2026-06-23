#include "minirpc/event_loop.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace minirpc {

void throw_system_error(const char* what) {
    throw std::system_error(errno, std::generic_category(), what);
}

void set_non_blocking(int fd) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags == -1) {
        throw_system_error("fcntl(F_GETFL)");
    }
    if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        throw_system_error("fcntl(F_SETFL)");
    }
}

EventLoop::EventLoop() {
    epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd_ == -1) {
        throw_system_error("epoll_create1");
    }

    event_fd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (event_fd_ == -1) {
        const int saved_errno = errno;
        ::close(epoll_fd_);
        errno = saved_errno;
        throw_system_error("eventfd");
    }

    add(event_fd_, EPOLLIN, [this](std::uint32_t) { drain_wakeup(); });
}

EventLoop::~EventLoop() {
    stop();
    if (event_fd_ != -1) {
        remove(event_fd_);
        ::close(event_fd_);
        event_fd_ = -1;
    }
    if (epoll_fd_ != -1) {
        ::close(epoll_fd_);
        epoll_fd_ = -1;
    }
}

void EventLoop::add(int fd, std::uint32_t events, EventHandler handler) {
    {
        std::lock_guard<std::mutex> lock(handlers_mutex_);
        handlers_[fd] = std::move(handler);
    }

    epoll_event event{};
    event.events = events;
    event.data.fd = fd;
    if (::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &event) == -1) {
        {
            std::lock_guard<std::mutex> lock(handlers_mutex_);
            handlers_.erase(fd);
        }
        throw_system_error("epoll_ctl(ADD)");
    }
}

void EventLoop::modify(int fd, std::uint32_t events) {
    epoll_event event{};
    event.events = events;
    event.data.fd = fd;
    if (::epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &event) == -1 && errno != ENOENT && errno != EBADF) {
        throw_system_error("epoll_ctl(MOD)");
    }
}

void EventLoop::remove(int fd) {
    ::epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    handlers_.erase(fd);
}

void EventLoop::post(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        tasks_.push(std::move(task));
    }
    wakeup();
}

void EventLoop::run() {
    stopped_ = false;
    std::vector<epoll_event> events(64);

    while (!stopped_) {
        drain_tasks();
        const int ready = ::epoll_wait(epoll_fd_, events.data(), static_cast<int>(events.size()), 1000);
        if (ready == -1) {
            if (errno == EINTR) {
                continue;
            }
            throw_system_error("epoll_wait");
        }

        for (int i = 0; i < ready; ++i) {
            EventHandler handler;
            {
                std::lock_guard<std::mutex> lock(handlers_mutex_);
                auto it = handlers_.find(events[static_cast<std::size_t>(i)].data.fd);
                if (it != handlers_.end()) {
                    handler = it->second;
                }
            }
            if (handler) {
                handler(events[static_cast<std::size_t>(i)].events);
            }
        }
    }

    drain_tasks();
}

void EventLoop::stop() {
    stopped_ = true;
    if (event_fd_ != -1) {
        wakeup();
    }
}

void EventLoop::wakeup() {
    const std::uint64_t value = 1;
    const auto written = ::write(event_fd_, &value, sizeof(value));
    if (written == -1 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EBADF) {
        throw_system_error("eventfd write");
    }
}

void EventLoop::drain_wakeup() {
    std::uint64_t value = 0;
    while (::read(event_fd_, &value, sizeof(value)) > 0) {
    }
}

void EventLoop::drain_tasks() {
    std::queue<std::function<void()>> local;
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        std::swap(local, tasks_);
    }

    while (!local.empty()) {
        auto task = std::move(local.front());
        local.pop();
        task();
    }
}

}  // namespace minirpc
