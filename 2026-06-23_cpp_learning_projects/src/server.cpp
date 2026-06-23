#include "minirpc/server.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

namespace minirpc {

TcpServer::TcpServer(EventLoop& loop, std::string host, std::uint16_t port)
    : loop_(loop),
      host_(std::move(host)),
      port_(port) {
}

TcpServer::~TcpServer() {
    stop();
}

void TcpServer::start() {
    if (started_) {
        return;
    }

    bind_and_listen();
    set_non_blocking(listen_fd_);

    loop_.add(listen_fd_, EPOLLIN, [this](std::uint32_t events) {
        if ((events & EPOLLIN) != 0) {
            accept_loop();
        }
    });

    started_ = true;
}

void TcpServer::stop() {
    if (!started_ && listen_fd_ == -1) {
        return;
    }

    close_listener();
    for (auto& item : sessions_) {
        item.second->close();
    }
    sessions_.clear();
    started_ = false;
}

void TcpServer::set_session_handler(SessionHandler handler) {
    session_handler_ = std::move(handler);
}

void TcpServer::set_message_handler(MessageHandler handler) {
    message_handler_ = std::move(handler);
}

void TcpServer::bind_and_listen() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listen_fd_ == -1) {
        throw_system_error("socket");
    }

    const int yes = 1;
    if (::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == -1) {
        close_listener();
        throw_system_error("setsockopt(SO_REUSEADDR)");
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
        close_listener();
        throw std::runtime_error("TcpServer only accepts IPv4 literal host addresses");
    }

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == -1) {
        close_listener();
        throw_system_error("bind");
    }

    if (::listen(listen_fd_, SOMAXCONN) == -1) {
        close_listener();
        throw_system_error("listen");
    }
}

void TcpServer::accept_loop() {
    while (true) {
        sockaddr_in peer{};
        socklen_t peer_len = sizeof(peer);
        const int fd = ::accept4(listen_fd_,
                                 reinterpret_cast<sockaddr*>(&peer),
                                 &peer_len,
                                 SOCK_CLOEXEC | SOCK_NONBLOCK);
        if (fd == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return;
            }
            if (errno == EINTR) {
                continue;
            }
            throw_system_error("accept4");
        }

        auto session = std::make_shared<Session>(loop_, fd);
        session->set_message_handler(message_handler_);
        session->set_close_handler([this](const Session::Ptr& closed) {
            sessions_.erase(closed->id());
        });
        sessions_[session->id()] = session;
        session->start();

        if (session_handler_) {
            session_handler_(session);
        }
    }
}

void TcpServer::close_listener() {
    if (listen_fd_ != -1) {
        loop_.remove(listen_fd_);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
}

}  // namespace minirpc
